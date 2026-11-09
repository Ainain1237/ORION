"""
tests/test_reinforcement_learning.py
"""

import pytest
from datetime import datetime, timezone

from mission.scenario import build_default_scenario
from mission.planner import GreedyPlanner, evaluate_schedule
from ai.reinforcement_learning import RLPlanner, ORIONEnv, QLearningAgent, ACTIONS
from spacecraft.satellite import Satellite

EPOCH = datetime(2026, 8, 16, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def scenario():
    return build_default_scenario(EPOCH, n_orbits=4.0, step_s=30)


class TestORIONEnv:
    def test_reset_returns_valid_state(self, scenario):
        timeline, opportunities, _ = scenario
        env = ORIONEnv(lambda: Satellite(), timeline, opportunities)
        state = env.reset()
        assert isinstance(state, tuple)
        assert len(state) == 5

    def test_episode_runs_to_completion(self, scenario):
        timeline, opportunities, _ = scenario
        env = ORIONEnv(lambda: Satellite(), timeline, opportunities)
        state = env.reset()
        steps = 0
        done = False
        while not done:
            _next, _r, done, _act = env.step(0)  # always IDLE
            steps += 1
        assert steps == len(timeline)


class TestQLearningAgent:
    def test_training_improves_average_reward(self, scenario):
        timeline, opportunities, _ = scenario
        env = ORIONEnv(lambda: Satellite(), timeline, opportunities)
        agent = QLearningAgent(seed=0)
        history = agent.train(env, episodes=60)
        early_avg = sum(history[:10]) / 10
        late_avg = sum(history[-10:]) / 10
        assert late_avg >= early_avg  # learning should not make things worse on average


class TestRLPlanner:
    def test_plan_produces_valid_schedule(self, scenario):
        timeline, opportunities, _ = scenario
        sat = Satellite()
        planner = RLPlanner(episodes=60, seed=0)
        choices = planner.plan(sat, timeline, opportunities)
        assert len(choices) == len(timeline)
        assert all(c in ACTIONS for c in choices)

    def test_rl_beats_or_matches_all_idle(self, scenario):
        timeline, opportunities, _ = scenario
        sat = Satellite()
        planner = RLPlanner(episodes=80, seed=1)
        choices = planner.plan(sat, timeline, opportunities)
        result = evaluate_schedule(sat, timeline, opportunities, choices)
        idle_result = evaluate_schedule(sat, timeline, opportunities,
                                         [choices[0].__class__.IDLE] * len(timeline))
        assert result.reward >= idle_result.reward


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
