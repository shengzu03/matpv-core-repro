"""Verify released audit files and optionally recomputed audit/cost outputs."""
import argparse
import hashlib
import json
from pathlib import Path

from audit_archived_predictions import compare

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "audits/jisa_20261110"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-output", type=Path)
    parser.add_argument("--cost-output", type=Path)
    args = parser.parse_args()
    manifest = json.loads((ARCHIVE / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        path = ROOT / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Missing or changed released file: " + name)
    print("AUDIT_RELEASE_FILES_OK", len(manifest["files"]))
    if args.prediction_output:
        old = json.loads((ARCHIVE / "expected/analysis_results.json").read_text(encoding="utf-8"))
        new = json.loads(args.prediction_output.read_text(encoding="utf-8"))
        for key in ("by_split", "by_cohort", "TON_stages"):
            compare(old[key], new[key], key)
        print("PREDICTION_OUTPUT_MATCHES_ARCHIVE")
    if args.cost_output:
        old = json.loads((ARCHIVE / "expected/cost_analysis.json").read_text(encoding="utf-8"))
        new = json.loads(args.cost_output.read_text(encoding="utf-8"))
        compare(old, new, "cost")
        print("COST_OUTPUT_MATCHES_ARCHIVE")


if __name__ == "__main__":
    main()
