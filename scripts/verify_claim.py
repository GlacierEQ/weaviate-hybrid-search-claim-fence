#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hybrid_search_evidence import Decision, HybridSearchEvidence, HybridSearchEvidenceRequest


def _payload() -> dict:
    return {
        "now": 100.0,
        "query": "Does the system preserve provenance?",
        "claim": "The system preserves source provenance.",
        "index_snapshot_digest": "a" * 64,
        "alpha": 0.5,
        "hits": [
            {
                "document_id": "d1",
                "source_uri": "https://example.org/a",
                "document_digest": "b" * 64,
                "bm25_score": 0.9,
                "vector_score": 0.8,
                "stance": "support",
                "evidence_strength": 0.9,
            },
            {
                "document_id": "d2",
                "source_uri": "https://example.net/b",
                "document_digest": "c" * 64,
                "bm25_score": 0.8,
                "vector_score": 0.9,
                "stance": "support",
                "evidence_strength": 0.8,
            },
        ],
        "policy": {
            "min_hybrid_score": 0.6,
            "min_supporting_hits": 2,
            "min_supporting_sources": 2,
            "min_support_weight": 1.2,
            "max_contradiction_weight": 0.3,
        },
    }


def main() -> int:
    engine = HybridSearchEvidence()
    baseline = engine.evaluate(HybridSearchEvidenceRequest("operate", _payload(), budget=4.0))
    if baseline.decision is not Decision.ALLOW:
        print(json.dumps(baseline.as_dict(), indent=2, sort_keys=True))
        return 2

    rebound_payload = _payload()
    rebound_payload["expected_evidence_digest"] = baseline.result["evidence_digest"]
    rebound = engine.evaluate(HybridSearchEvidenceRequest("operate", rebound_payload, budget=4.0))

    contradicted_payload = _payload()
    contradicted_payload["hits"].append(
        {
            "document_id": "d3",
            "source_uri": "https://example.com/c",
            "document_digest": "d" * 64,
            "bm25_score": 0.95,
            "vector_score": 0.95,
            "stance": "contradict",
            "evidence_strength": 0.9,
        }
    )
    contradicted = engine.evaluate(
        HybridSearchEvidenceRequest("operate", contradicted_payload, budget=4.0)
    )

    tampered_payload = deepcopy(rebound_payload)
    tampered_payload["hits"][0]["document_digest"] = "e" * 64
    tampered = engine.evaluate(HybridSearchEvidenceRequest("operate", tampered_payload, budget=4.0))

    print(
        json.dumps(
            {"baseline": baseline.as_dict(), "rebound": rebound.as_dict(), "contradicted": contradicted.as_dict(), "tampered": tampered.as_dict()},
            indent=2,
            sort_keys=True,
        )
    )
    if rebound.decision is not Decision.ALLOW:
        return 3
    if contradicted.decision is not Decision.REFUSE or "contradiction_weight_exceeded" not in contradicted.reasons:
        return 4
    if tampered.decision is not Decision.REFUSE or "expected_evidence_digest_mismatch" not in tampered.reasons:
        return 5
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
