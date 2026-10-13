"""Entry point. Run from the LangSplatV2 root:

    python -m simulator.main --config simulator/configs/silf.yaml
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from simulator.simulator import run_simulator


def parse_args():
    parser = argparse.ArgumentParser(description="SILF ASIC-PIM simulator")
    parser.add_argument(
        "--config",
        type=str,
        default="simulator/configs/silf.yaml",
        help="yaml config",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    stats = run_simulator(args.config)
    if hasattr(stats, "total_cycles"):
        print(f"[main] done. total_cycles={stats.total_cycles:.1f} fps={stats.fps:.2f}")
    elif isinstance(stats, dict) and stats.get("mode") == "multi_scene":
        avg = stats.get("average_fps")
        extra = f", average_fps={avg:.2f}" if avg else ""
        print(f"[main] done. scenes={len(stats.get('scenes', []))}{extra}")
    else:
        print("[main] done.")


if __name__ == "__main__":
    main()
