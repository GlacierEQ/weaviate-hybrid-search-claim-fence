from hybrid_search_evidence import Decision, HybridSearchEvidence, HybridSearchEvidenceRequest

SNAPSHOT = "a" * 64
D1 = "b" * 64
D2 = "c" * 64
D3 = "d" * 64


def hits() -> list[dict]:
    return [
        {
            "document_id": "doc-1",
            "source_uri": "https://source.example/a",
            "document_digest": D1,
            "bm25_score": 0.9,
            "vector_score": 0.8,
            "stance": "support",
            "evidence_strength": 0.9,
        },
        {
            "document_id": "doc-2",
            "source_uri": "https://source.example/b",
            "document_digest": D2,
            "bm25_score": 0.8,
            "vector_score": 0.9,
            "stance": "support",
            "evidence_strength": 0.8,
        },
        {
            "document_id": "doc-3",
            "source_uri": "https://source.example/c",
            "document_digest": D3,
            "bm25_score": 0.5,
            "vector_score": 0.5,
            "stance": "neutral",
            "evidence_strength": 1.0,
        },
    ]


def payload(**changes) -> dict:
    value = {
        "now": 100.0,
        "query": "What supports the claim?",
        "claim": "The system preserves source provenance.",
        "index_snapshot_digest": SNAPSHOT,
        "alpha": 0.5,
        "hits": hits(),
        "policy": {
            "min_hybrid_score": 0.6,
            "min_supporting_hits": 2,
            "min_supporting_sources": 2,
            "min_support_weight": 1.2,
            "max_contradiction_weight": 0.3,
        },
    }
    value.update(changes)
    return value


def evaluate(value=None, *, budget=4.0, not_after=None):
    return HybridSearchEvidence().evaluate(
        HybridSearchEvidenceRequest("claim-1", value or payload(), budget=budget, not_after=not_after)
    )


def test_diverse_supporting_hybrid_hits_allow():
    receipt = evaluate()
    assert receipt.decision is Decision.ALLOW
    assert receipt.reasons == ("claim_supported_by_hybrid_retrieval_evidence",)
    assert receipt.metrics["supporting_hit_count"] == 2
    assert receipt.metrics["supporting_source_count"] == 2
    assert receipt.metrics["support_weight"] > 1.2
    assert len(receipt.result["evidence_digest"]) == 64


def test_hybrid_ranking_uses_declared_alpha():
    receipt = evaluate(payload(alpha=0.75))
    ranked = receipt.result["manifest"]["ranked_hits"]
    assert ranked[0]["document_id"] == "doc-2"
    assert ranked[0]["hybrid_score"] > ranked[1]["hybrid_score"]


def test_source_diversity_is_not_same_as_hit_count():
    changed = hits()
    changed[1]["source_uri"] = changed[0]["source_uri"]
    receipt = evaluate(payload(hits=changed))
    assert receipt.decision is Decision.REFUSE
    assert "insufficient_supporting_sources" in receipt.reasons


def test_high_contradiction_weight_refuses():
    changed = hits()
    changed.append(
        {
            "document_id": "doc-4",
            "source_uri": "https://source.example/d",
            "document_digest": "e" * 64,
            "bm25_score": 0.95,
            "vector_score": 0.95,
            "stance": "contradict",
            "evidence_strength": 0.9,
        }
    )
    receipt = evaluate(payload(hits=changed))
    assert receipt.decision is Decision.REFUSE
    assert "contradiction_weight_exceeded" in receipt.reasons


def test_low_scoring_support_does_not_count():
    changed = hits()
    changed[1]["bm25_score"] = 0.1
    changed[1]["vector_score"] = 0.1
    receipt = evaluate(payload(hits=changed))
    assert receipt.decision is Decision.REFUSE
    assert "insufficient_supporting_hits" in receipt.reasons
    assert "insufficient_supporting_sources" in receipt.reasons


def test_expected_evidence_digest_binds_snapshot_and_hits():
    baseline = evaluate()
    expected = baseline.result["evidence_digest"]
    verified = evaluate(payload(expected_evidence_digest=expected))
    assert verified.decision is Decision.ALLOW

    changed = hits()
    changed[0]["document_digest"] = "f" * 64
    refused = evaluate(payload(hits=changed, expected_evidence_digest=expected))
    assert refused.decision is Decision.REFUSE
    assert "expected_evidence_digest_mismatch" in refused.reasons


def test_snapshot_identity_change_changes_evidence_identity():
    first = evaluate()
    second = evaluate(payload(index_snapshot_digest="f" * 64))
    assert first.result["evidence_digest"] != second.result["evidence_digest"]


def test_duplicate_document_ids_refuse():
    changed = hits()
    changed[1]["document_id"] = "doc-1"
    receipt = evaluate(payload(hits=changed))
    assert receipt.decision is Decision.REFUSE
    assert "duplicate_document_id" in receipt.reasons


def test_scores_and_strength_are_strict_unit_intervals():
    changed = hits()
    changed[0]["bm25_score"] = 1.1
    receipt = evaluate(payload(hits=changed))
    assert receipt.decision is Decision.REFUSE
    assert "hit_0_bm25_score_invalid" in receipt.reasons


def test_request_expiry_refuses():
    receipt = evaluate(not_after=99.0)
    assert receipt.decision is Decision.REFUSE
    assert "request_expired" in receipt.reasons


def test_work_budget_refuses_before_hit_normalization():
    changed = hits() * 40
    # Give every duplicate a distinct id only if normalization happened; budget must stop first.
    receipt = evaluate(payload(hits=changed), budget=0.6)
    assert receipt.decision is Decision.REFUSE
    assert "work_budget_exceeded" in receipt.reasons
    assert receipt.metrics["eligible_hit_count"] == 0
