"""Behavioral scaffold tests for HybridSearchClaimFence.

Filling AI: keep allow/refuse polarity; deepen assertions as the real algorithm lands.
"""
from __future__ import annotations

from hybrid_search_claim_fence import Decision, HybridSearchClaimFence, HybridSearchClaimFenceRequest


def test_allow_path_returns_digest() -> None:
    mech = HybridSearchClaimFence()
    receipt = mech.evaluate(
        HybridSearchClaimFenceRequest(subject_id="a", payload={"x": 1}, budget=1.0)
    )
    assert receipt.decision is Decision.ALLOW
    assert len(receipt.digest) == 64
    assert receipt.metrics.get("scaffold") is True


def test_refuse_missing_subject() -> None:
    mech = HybridSearchClaimFence()
    receipt = mech.evaluate(HybridSearchClaimFenceRequest(subject_id="  ", payload={}, budget=1.0))
    assert receipt.decision is Decision.REFUSE
    assert "subject_id_missing" in receipt.reasons


def test_refuse_non_positive_budget() -> None:
    mech = HybridSearchClaimFence()
    receipt = mech.evaluate(HybridSearchClaimFenceRequest(subject_id="a", payload={}, budget=0.0))
    assert receipt.decision is Decision.REFUSE
    assert "budget_non_positive" in receipt.reasons


def test_different_payloads_different_digests() -> None:
    mech = HybridSearchClaimFence()
    a = mech.evaluate(HybridSearchClaimFenceRequest(subject_id="a", payload={"n": 1}, budget=1.0))
    b = mech.evaluate(HybridSearchClaimFenceRequest(subject_id="a", payload={"n": 2}, budget=1.0))
    assert a.digest != b.digest
