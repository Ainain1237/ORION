"""
dashboard/build_dashboard.py

Generates a self-contained mission_control.html by embedding a fresh
simulation run (orbit + telemetry + AI decisions + anomalies) into
dashboard/template.html. The output is a single file with no server
required — open it directly in a browser.

Run:
    python -m dashboard.build_dashboard [planner_name]
Output:
    dashboard/mission_control.html
"""

import os
import sys
import json

from simulation.export_dashboard_data import build_dashboard_data

HERE = os.path.dirname(__file__)


def build(planner_name: str = "greedy", n_orbits: float = 6.0, step_s: float = 30.0):
    data = build_dashboard_data(planner_name=planner_name, n_orbits=n_orbits, step_s=step_s)

    template_path = os.path.join(HERE, "template.html")
    with open(template_path, "r") as f:
        template = f.read()

    json_str = json.dumps(data)
    html = template.replace("__ORION_DATA__", json_str)

    out_path = os.path.join(HERE, "mission_control.html")
    with open(out_path, "w") as f:
        f.write(html)

    print(f"Saved: {out_path}  ({os.path.getsize(out_path)/1024:.0f} KB, "
          f"{len(data['steps'])} steps, planner={planner_name})")
    return out_path


if __name__ == "__main__":
    planner = sys.argv[1] if len(sys.argv) > 1 else "greedy"
    build(planner_name=planner)
