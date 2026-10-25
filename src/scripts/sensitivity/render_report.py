"""Summarize a completed run at its CAL threshold."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from report_results import main
if __name__=='__main__':main(sys.argv[1:])
