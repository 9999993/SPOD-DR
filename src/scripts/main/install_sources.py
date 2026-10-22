"""Compatibility check for bundled sources; no installation or file copying."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training_resources import inspect_resources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets-root', help='Deprecated; bundled sources are used directly')
    args = parser.parse_args()
    if args.assets_root:
        parser.error('--assets-root is no longer needed; use training.py check for resource checks')
    result = inspect_resources(family='native')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result['issues']: raise SystemExit(1)


if __name__ == '__main__': main()
