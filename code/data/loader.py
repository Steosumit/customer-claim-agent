import csv
import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ClaimRow:
    row_index: int
    user_id: str
    image_paths: List[str]
    user_claim: str
    claim_object: str


@dataclass
class ClaimContext:
    row: ClaimRow
    user_history: dict
    evidence_requirements: List[dict]
    image_abs_paths: List[str]


def load_csv(path: str) -> List[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_claims(path: str) -> List[ClaimRow]:
    rows = load_csv(path)
    claims = []
    for i, r in enumerate(rows):
        image_paths = [p.strip() for p in r["image_paths"].split(";") if p.strip()]
        claims.append(ClaimRow(
            row_index=i,
            user_id=r["user_id"],
            image_paths=image_paths,
            user_claim=r["user_claim"],
            claim_object=r["claim_object"],
        ))
    return claims


def load_user_history(path: str) -> dict:
    rows = load_csv(path)
    return {r["user_id"]: r for r in rows}


def load_evidence_requirements(path: str) -> List[dict]:
    return load_csv(path)


def resolve_image_paths(image_paths: List[str], repo_root: str, dataset_dir: str) -> List[str]:
    resolved = []
    for p in image_paths:
        if os.path.isabs(p):
            resolved.append(p)
        else:
            resolved.append(os.path.join(repo_root, dataset_dir, p))
    return resolved


def build_context(
    claim: ClaimRow,
    user_history: dict,
    evidence_requirements: List[dict],
    repo_root: str,
    dataset_dir: str,
) -> ClaimContext:
    history = user_history.get(claim.user_id, {})
    relevant_reqs = [
        r for r in evidence_requirements
        if r["claim_object"] == claim.claim_object or r["claim_object"] == "all"
    ]
    image_abs = resolve_image_paths(claim.image_paths, repo_root, dataset_dir)
    return ClaimContext(
        row=claim,
        user_history=history,
        evidence_requirements=relevant_reqs,
        image_abs_paths=image_abs,
    )
