import sys
import os
import time
import json

sys.path.insert(0, os.path.dirname(__file__))

from config import REPO_ROOT, DATASET_DIR, OUTPUT_PATH
from data.loader import load_claims, load_user_history, load_evidence_requirements, build_context
from data.writer import write_output
from crews.jury_crew import run_pipeline


def main():
    claims_path = os.path.join(REPO_ROOT, DATASET_DIR, "claims.csv")
    history_path = os.path.join(REPO_ROOT, DATASET_DIR, "user_history.csv")
    reqs_path = os.path.join(REPO_ROOT, DATASET_DIR, "evidence_requirements.csv")
    output_path = os.path.join(REPO_ROOT, OUTPUT_PATH)

    print("Loading data...")
    claims = load_claims(claims_path)
    history = load_user_history(history_path)
    reqs = load_evidence_requirements(reqs_path)
    print(f"Loaded {len(claims)} claims, {len(history)} user histories, {len(reqs)} requirements")

    results = []
    start_time = time.time()

    for i, claim in enumerate(claims):
        print(f"\n--- [{i+1}/{len(claims)}] Processing {claim.user_id} ({claim.claim_object}) ---")
        claim_start = time.time()
        context = build_context(claim, history, reqs, REPO_ROOT, DATASET_DIR)

        try:
            result = run_pipeline(claim, context)
            result.setdefault("user_id", claim.user_id)
            result.setdefault("image_paths", ";".join(claim.image_paths))
            result.setdefault("user_claim", claim.user_claim)
            result.setdefault("claim_object", claim.claim_object)
            results.append(result)
            elapsed = time.time() - claim_start
            print(f"  -> {result.get('claim_status', 'unknown')} ({elapsed:.1f}s)")
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "user_id": claim.user_id,
                "image_paths": ";".join(claim.image_paths),
                "user_claim": claim.user_claim,
                "claim_object": claim.claim_object,
                "evidence_standard_met": "false",
                "evidence_standard_met_reason": f"Processing error: {e}",
                "risk_flags": "none",
                "issue_type": "unknown",
                "object_part": "unknown",
                "claim_status": "not_enough_information",
                "claim_status_justification": f"Error processing claim: {e}",
                "supporting_image_ids": "none",
                "valid_image": "false",
                "severity": "unknown",
            })

    write_output(results, output_path)
    total_time = time.time() - start_time
    print(f"\nDone. Processed {len(results)} claims in {total_time:.1f}s ({total_time/len(results):.1f}s avg)")


if __name__ == "__main__":
    main()
