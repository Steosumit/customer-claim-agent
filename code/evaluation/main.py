import sys
import os
import json
import time
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import REPO_ROOT, DATASET_DIR
from data.loader import load_claims, load_user_history, load_evidence_requirements, build_context
from data.writer import OUTPUT_COLUMNS
from crews.jury_crew import run_pipeline

COMPARE_FIELDS = [
    "evidence_standard_met", "issue_type", "object_part",
    "claim_status", "severity", "valid_image",
]


def evaluate(max_rows=None):
    sample_path = os.path.join(REPO_ROOT, DATASET_DIR, "sample_claims.csv")
    history_path = os.path.join(REPO_ROOT, DATASET_DIR, "user_history.csv")
    reqs_path = os.path.join(REPO_ROOT, DATASET_DIR, "evidence_requirements.csv")

    claims = load_claims(sample_path)
    history = load_user_history(history_path)
    reqs = load_evidence_requirements(reqs_path)

    if max_rows:
        claims = claims[:max_rows]

    results = []
    correct = {f: 0 for f in COMPARE_FIELDS}
    total = 0
    start = time.time()

    for i, claim in enumerate(claims):
        print(f"\n--- Eval [{i+1}/{len(claims)}] {claim.user_id} ---")
        ctx = build_context(claim, history, reqs, REPO_ROOT, DATASET_DIR)

        try:
            result = run_pipeline(claim, ctx)
        except Exception as e:
            print(f"  ERROR: {e}")
            result = {"claim_status": "not_enough_information", "issue_type": "unknown",
                      "object_part": "unknown", "severity": "unknown", "valid_image": "false",
                      "evidence_standard_met": "false"}

        # Load expected from CSV
        expected = {}
        with open(sample_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for j, row in enumerate(reader):
                if j == i:
                    expected = row
                    break

        for field in COMPARE_FIELDS:
            got = str(result.get(field, "")).lower().strip()
            exp = str(expected.get(field, "")).lower().strip()
            if got == exp:
                correct[field] += 1
            else:
                print(f"  {field}: got={got} expected={exp}")
        total += 1
        results.append(result)

    elapsed = time.time() - start
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    for field in COMPARE_FIELDS:
        acc = correct[field] / total * 100 if total else 0
        print(f"  {field}: {correct[field]}/{total} ({acc:.1f}%)")
    overall = sum(correct.values()) / (total * len(COMPARE_FIELDS)) * 100
    print(f"\n  Overall accuracy: {overall:.1f}%")
    print(f"  Time: {elapsed:.1f}s ({elapsed/total:.1f}s/row)")

    report = {
        "total_rows": total,
        "accuracy_per_field": {f: correct[f] / total for f in COMPARE_FIELDS},
        "overall_accuracy": overall,
        "time_seconds": elapsed,
        "results": results,
    }
    report_path = os.path.join(os.path.dirname(__file__), "eval_results.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nDetailed results saved to {report_path}")
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()
    evaluate(max_rows=args.max_rows)
