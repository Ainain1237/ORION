"""
ai/reinforcement_learning.py

A reinforcement-learning mission planner, framed as a Gym-style
environment so it's a drop-in replacement for gymnasium/stable-baselines3
later if you want deeper function approximation (a small PyTorch policy
network is a natural next step — the environment interface below is
exactly what that would plug into).

For now this uses tabular Q-learning with a discretized state space
(battery bin x storage bin x thermal-safe flag x opportunity flags),
which is fast to train (no GPU/heavy deps needed) and transparent to
inspect — you can print the learned Q-table and literally read off the
policy, which is valuable for a research report.

    ORIONEnv        — Gym-style environment: reset()/step(action)
    QLearningAgent   — tabular Q-learning trainer
    RLPlanner         — wraps a trained QLearningAgent behind the same
                         .plan(satellite, timeline, opportunities) interface
                         as the other planners, so Experiment 1 can compare
                         it directly against Greedy/GA/IP.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import numpy as np

from mission.activities import ActivityType, ACTIVITY_PROFILES
from mission.constraints import check_activity_feasible
from mission.planner import TimeStep, opportunities_active_at
from spacecraft.satellite import Satellite


ACTIONS = [ActivityType.IDLE, ActivityType.CHARGE, ActivityType.IMAGE, ActivityType.DOWNLINK]


def _discretize_state(sat: Satellite, has_image_opp: bool, has_downlink_opp: bool,
                       n_battery_bins: int = 5, n_storage_bins: int = 3) -> tuple:
    battery_bin = min(n_battery_bins - 1, int(sat.battery.soc_pct / 100 * n_battery_bins))
    storage_bin = min(n_storage_bins - 1, int(sat.storage.used_pct / 100 * n_storage_bins))
    thermal_safe = sat.thermal.payload_safe()
    return (battery_bin, storage_bin, thermal_safe, has_image_opp, has_downlink_opp)


class ORIONEnv:
    """
    Gym-style environment: one episode = one full mission timeline.
    observation = discretized state tuple (see _discretize_state)
    action = index into ACTIONS
    reward = mission reward for this step (image capture, data
             downlinked, minus a penalty for constraint violations)
    """

    def __init__(self, satellite_factory, timeline: list[TimeStep], opportunities: list):
        self.satellite_factory = satellite_factory  # callable -> fresh Satellite
        self.timeline = timeline
        self.opportunities = opportunities
        self.sat: Satellite | None = None
        self.i = 0

    def reset(self):
        self.sat = self.satellite_factory()
        self.i = 0
        return self._obs()

    def _obs(self):
        step = self.timeline[self.i]
        has_img = bool(opportunities_active_at(self.opportunities, step.t, ActivityType.IMAGE))
        has_dl = bool(opportunities_active_at(self.opportunities, step.t, ActivityType.DOWNLINK))
        return _discretize_state(self.sat, has_img, has_dl)

    def step(self, action_idx: int):
        step = self.timeline[self.i]
        requested = ACTIONS[action_idx]

        has_img = bool(opportunities_active_at(self.opportunities, step.t, ActivityType.IMAGE))
        has_dl = bool(opportunities_active_at(self.opportunities, step.t, ActivityType.DOWNLINK))

        # imaging/downlink only meaningful (and rewarded) inside a real opportunity
        activity = requested
        reward = 0.0
        if requested == ActivityType.IMAGE and not has_img:
            activity = ActivityType.IDLE
        if requested == ActivityType.DOWNLINK and not has_dl:
            activity = ActivityType.IDLE

        violation = check_activity_feasible(self.sat, activity, step.dt_s, step.in_eclipse)
        if violation is not None:
            reward -= 15.0
            activity = ActivityType.IDLE

        profile = ACTIVITY_PROFILES[activity]
        state = self.sat.step(step.t, step.dt_s, step.in_eclipse, profile, activity.value,
                               strict=False)

        if activity == ActivityType.IMAGE and state.fault == "":
            reward += 10.0
        if activity == ActivityType.DOWNLINK:
            reward += 0.2 * profile.downlink_rate_mbps * step.dt_s / 8.0

        self.i += 1
        done = self.i >= len(self.timeline)
        obs = self._obs() if not done else None
        return obs, reward, done, activity


class QLearningAgent:
    def __init__(self, alpha: float = 0.15, gamma: float = 0.95,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.05, seed: int = 0):
        self.q: dict = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.rng = random.Random(seed)

    def _q(self, state) -> np.ndarray:
        if state not in self.q:
            self.q[state] = np.zeros(len(ACTIONS))
        return self.q[state]

    def select_action(self, state, epsilon: float) -> int:
        if self.rng.random() < epsilon:
            return self.rng.randrange(len(ACTIONS))
        return int(np.argmax(self._q(state)))

    def update(self, state, action, reward, next_state, done):
        q_sa = self._q(state)[action]
        target = reward if done else reward + self.gamma * np.max(self._q(next_state))
        self._q(state)[action] += self.alpha * (target - q_sa)

    def train(self, env: ORIONEnv, episodes: int = 150):
        history = []
        for ep in range(episodes):
            epsilon = self.epsilon_start + (self.epsilon_end - self.epsilon_start) * (ep / max(1, episodes - 1))
            state = env.reset()
            total_reward = 0.0
            done = False
            while not done:
                action = self.select_action(state, epsilon)
                next_state, reward, done, _activity = env.step(action)
                self.update(state, action, reward, next_state if next_state else state, done)
                total_reward += reward
                state = next_state if next_state else state
            history.append(total_reward)
        return history

    def greedy_policy(self, state) -> int:
        return int(np.argmax(self._q(state)))


class RLPlanner:
    """
    Wraps a trained QLearningAgent behind the same `.plan(satellite,
    timeline, opportunities) -> list[ActivityType]` interface as the
    other planners, so it can be compared directly in Experiment 1.
    Training happens once per scenario (episodes over the same
    timeline/opportunities), then the greedy policy is extracted.
    """
    name = "Reinforcement Learning (Q-learning)"

    def __init__(self, episodes: int = 150, seed: int = 0):
        self.episodes = episodes
        self.seed = seed
        self.agent: QLearningAgent | None = None
        self.training_history: list = []

    def plan(self, satellite: Satellite, timeline: list[TimeStep], opportunities: list) -> list[ActivityType]:
        def factory():
            return copy.deepcopy(satellite)

        env = ORIONEnv(factory, timeline, opportunities)
        self.agent = QLearningAgent(seed=self.seed)
        self.training_history = self.agent.train(env, episodes=self.episodes)

        # extract the greedy (learned) policy as one deterministic pass
        state = env.reset()
        choices = []
        done = False
        while not done:
            action = self.agent.greedy_policy(state)
            next_state, _reward, done, activity = env.step(action)
            choices.append(activity)
            state = next_state if next_state else state
        return choices
