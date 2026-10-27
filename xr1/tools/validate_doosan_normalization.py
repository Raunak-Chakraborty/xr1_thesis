"""Validate converter-produced Doosan statistics without ML dependencies."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mibot.utils.doosan_profile import resolve_doosan_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("statistics", help="dataset-derived normalization JSON")
    parser.add_argument("--use-wrench", action="store_true")
    args = parser.parse_args()
    try:
        profile = resolve_doosan_data({
            "normalization_path": args.statistics,
            "use_wrench": args.use_wrench,
            "action_length": 30,
        })
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f"valid Doosan normalization: use_wrench={profile['use_wrench']}, "
          "state=(1,60), action=(30,60), action mask=0:7")


if __name__ == "__main__":
    main()
