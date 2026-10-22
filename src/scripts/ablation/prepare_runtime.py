"""Prepare user manifests through the unified training entry."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training_pipeline import main
if __name__ == '__main__': main(['prepare', *sys.argv[1:]])
