"""Deterministic evidence verification for normalized hybrid-search results.

A provider adapter owns retrieval. This kernel receives normalized sparse/vector
scores plus explicit evidence stance and decides whether a claim has enough
retrieval support to pass a declared evidence policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")


def _digest(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class Decision(str, Enum):
    ALLOW = "ALLOW"
    REFUSE = "REFUSE"


@dataclass(frozen=True)
class HybridSearchEvidenceRequest:
    subject_id: str
    payload: dict[str, Any] = field(default_factory=dict)
    budget: float = 4.0
    not_after: float | None = None


@dataclass(frozen=True)
class HybridSearchEvidenceReceipt:
    decision: Decision
    reasons: tuple[str, ...]
    digest: str
    metrics: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "reasons": list(self.reasons),
            "digest": self.digest,
            "metrics": self.metrics,
            "result": self.result,
        }


class HybridSearchEvidence:
    """Rank normalized hybrid hits and enforce evidence sufficiency."""

    MAX_HITS = 512
    MAX_TEXT = 16_384
    MAX_INPUT_CHARS = 2_000_000
    VALID_PAYLOAD_KEYS = frozenset(
        {
            "now",
            "query",
            "claim",
            "index_snapshot_digest",
            "alpha",
            "hits",
            "policy",
            "expected_evidence_digest",
        }
    )
    HIT_KEYS = frozenset(
        {
            "document_id",
            "source_uri",
            "document_digest",
            "bm25_score",
            "vector_score",
            "stance",
            "evidence_strength",
        }
    )
    POLICY_KEYS = frozenset(
        {
            "min_hybrid_score",
            "min_supporting_hits",
            "min_supporting_sources",
            "min_support_weight",
            "max_contradiction_weight",
        }
    )
    STANCES = frozenset({"support", "contradict", "neutral"})
    BASE_WORK = 0.5
    HIT_WORK = 0.01

    @classmethod
    def _text(cls, value: Any, label: str) -> str:
        if not isinstance(value, str):
            raise ValueError(f"{label}_type_invalid")
        value = value.strip()
        if not value:
            raise ValueError(f"{label}_missing")
        if len(value) > cls.MAX_TEXT:
            raise ValueError(f"{label}_too_long")
        if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in value):
            raise ValueError(f"{label}_control_character")
        return value

    @staticmethod
    def _unit(value: Any, label: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label}_invalid")
        value = float(value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{label}_invalid")
        return value

    @staticmethod
    def _positive(value: Any, label: str, *, minimum: float = 0.0) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label}_invalid")
        value = float(value)
        if not math.isfinite(value) or value < minimum:
            raise ValueError(f"{label}_invalid")
        return value

    @staticmethod
    def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{label}_invalid")
        return value

    @staticmethod
    def _sha(value: Any, label: str) -> str:
        if not isinstance(value, str):
            raise ValueError(f"{label}_type_invalid")
        value = value.lower()
        if not _SHA256.fullmatch(value):
            raise ValueError(f"{label}_invalid")
        return value

    @classmethod
    def _policy(cls, raw: Any) -> dict[str, Any]:
        if not isinstance(raw, Mapping):
            raise ValueError("policy_missing")
        unknown = set(raw) - cls.POLICY_KEYS
        if unknown:
            raise ValueError("policy_keys_unknown:" + ",".join(sorted(unknown)))
        return {
            "min_hybrid_score": cls._unit(raw.get("min_hybrid_score", 0.5), "policy_min_hybrid_score"),
            "min_supporting_hits": cls._integer(raw.get("min_supporting_hits", 2), "policy_min_supporting_hits", minimum=1),
            "min_supporting_sources": cls._integer(raw.get("min_supporting_sources", 2), "policy_min_supporting_sources", minimum=1),
            "min_support_weight": cls._positive(raw.get("min_support_weight", 1.0), "policy_min_support_weight", minimum=0.0),
            "max_contradiction_weight": cls._positive(raw.get("max_contradiction_weight", 0.5), "policy_max_contradiction_weight", minimum=0.0),
        }

    @classmethod
    def _hit(cls, raw: Any, index: int, alpha: float) -> dict[str, Any]:
        if not isinstance(raw, Mapping):
            raise ValueError(f"hit_{index}_not_object")
        unknown = set(raw) - cls.HIT_KEYS
        if unknown:
            raise ValueError(f"hit_{index}_keys_unknown:" + ",".join(sorted(unknown)))
        stance = cls._text(raw.get("stance"), f"hit_{index}_stance").lower()
        if stance not in cls.STANCES:
            raise ValueError(f"hit_{index}_stance_invalid")
        bm25 = cls._unit(raw.get("bm25_score"), f"hit_{index}_bm25_score")
        vector = cls._unit(raw.get("vector_score"), f"hit_{index}_vector_score")
        strength = cls._unit(raw.get("evidence_strength"), f"hit_{index}_evidence_strength")
        hybrid = (1.0 - alpha) * bm25 + alpha * vector
        return {
            "document_id": cls._text(raw.get("document_id"), f"hit_{index}_document_id"),
            "source_uri": cls._text(raw.get("source_uri"), f"hit_{index}_source_uri"),
            "document_digest": cls._sha(raw.get("document_digest"), f"hit_{index}_document_digest"),
            "bm25_score": bm25,
            "vector_score": vector,
            "hybrid_score": hybrid,
            "stance": stance,
            "evidence_strength": strength,
            "weighted_evidence": hybrid * strength,
        }

    def evaluate(self, req: HybridSearchEvidenceRequest) -> HybridSearchEvidenceReceipt:
        reasons: list[str] = []
        try:
            subject_id = self._text(req.subject_id, "subject_id")
        except ValueError as exc:
            subject_id = ""
            reasons.append(str(exc))
        try:
            budget = self._positive(req.budget, "budget", minimum=0.001)
        except ValueError as exc:
            budget = 0.0
            reasons.append(str(exc))
        if not isinstance(req.payload, Mapping):
            payload: Mapping[str, Any] = {}
            reasons.append("payload_not_object")
        else:
            payload = req.payload
            unknown = set(payload) - self.VALID_PAYLOAD_KEYS
            if unknown:
                reasons.append("payload_keys_unknown:" + ",".join(sorted(unknown)))

        if req.not_after is not None:
            try:
                now = self._positive(payload.get("now"), "now")
                not_after = self._positive(req.not_after, "not_after")
                if now > not_after:
                    reasons.append("request_expired")
            except ValueError as exc:
                reasons.append(str(exc))

        result: dict[str, Any] = {}
        work_units = self.BASE_WORK
        try:
            query = self._text(payload.get("query"), "query")
            claim = self._text(payload.get("claim"), "claim")
            snapshot = self._sha(payload.get("index_snapshot_digest"), "index_snapshot_digest")
            alpha = self._unit(payload.get("alpha", 0.5), "alpha")
            policy = self._policy(payload.get("policy"))
            hits_raw = payload.get("hits")
            if not isinstance(hits_raw, list):
                raise ValueError("hits_missing")
            if len(hits_raw) > self.MAX_HITS:
                raise ValueError("hits_over_limit")
            work_units += len(hits_raw) * self.HIT_WORK
            if work_units > budget:
                reasons.append("work_budget_exceeded")
            else:
                hits = [self._hit(raw, index, alpha) for index, raw in enumerate(hits_raw)]
                if len({hit["document_id"] for hit in hits}) != len(hits):
                    raise ValueError("duplicate_document_id")
                hits.sort(key=lambda hit: (-hit["hybrid_score"], hit["document_id"]))
                eligible = [hit for hit in hits if hit["hybrid_score"] >= policy["min_hybrid_score"]]
                supports = [hit for hit in eligible if hit["stance"] == "support"]
                contradictions = [hit for hit in eligible if hit["stance"] == "contradict"]
                support_sources = {hit["source_uri"] for hit in supports}
                support_weight = sum(hit["weighted_evidence"] for hit in supports)
                contradiction_weight = sum(hit["weighted_evidence"] for hit in contradictions)

                if len(supports) < policy["min_supporting_hits"]:
                    reasons.append("insufficient_supporting_hits")
                if len(support_sources) < policy["min_supporting_sources"]:
                    reasons.append("insufficient_supporting_sources")
                if support_weight < policy["min_support_weight"]:
                    reasons.append("insufficient_support_weight")
                if contradiction_weight > policy["max_contradiction_weight"]:
                    reasons.append("contradiction_weight_exceeded")

                evidence_manifest = {
                    "schema": "glaciereq.hybrid-search-evidence.v1",
                    "query": query,
                    "claim": claim,
                    "query_digest": _digest(query),
                    "claim_digest": _digest(claim),
                    "index_snapshot_digest": snapshot,
                    "alpha": alpha,
                    "policy": policy,
                    "ranked_hits": hits,
                }
                evidence_digest = _digest(evidence_manifest)
                expected = payload.get("expected_evidence_digest")
                if expected is not None:
                    expected = self._sha(expected, "expected_evidence_digest")
                    if expected != evidence_digest:
                        reasons.append("expected_evidence_digest_mismatch")
                result = {
                    "manifest": evidence_manifest,
                    "evidence_digest": evidence_digest,
                    "eligible_hit_count": len(eligible),
                    "supporting_hit_count": len(supports),
                    "supporting_source_count": len(support_sources),
                    "support_weight": support_weight,
                    "contradiction_weight": contradiction_weight,
                }
        except ValueError as exc:
            reasons.append(str(exc))

        decision = Decision.REFUSE if reasons else Decision.ALLOW
        if not reasons:
            reasons = ["claim_supported_by_hybrid_retrieval_evidence"]
        metrics = {
            "work_units": work_units,
            "budget_units": budget,
            "eligible_hit_count": result.get("eligible_hit_count", 0),
            "supporting_hit_count": result.get("supporting_hit_count", 0),
            "supporting_source_count": result.get("supporting_source_count", 0),
            "support_weight": result.get("support_weight", 0.0),
            "contradiction_weight": result.get("contradiction_weight", 0.0),
        }
        digest = _digest(
            {
                "subject_id": subject_id,
                "decision": decision.value,
                "reasons": reasons,
                "result": result,
                "metrics": metrics,
            }
        )
        return HybridSearchEvidenceReceipt(decision, tuple(reasons), digest, metrics, result)


def _read_input(path: str | None) -> str:
    limit = HybridSearchEvidence.MAX_INPUT_CHARS
    if path:
        source = Path(path)
        if source.stat().st_size > limit * 4:
            raise ValueError("input_too_large")
        raw = source.read_text(encoding="utf-8")
    else:
        raw = sys.stdin.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("input_too_large")
    return raw


def cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify claim support from normalized hybrid-search evidence.")
    parser.add_argument("--input", "-i", help="request JSON file; defaults to stdin")
    args = parser.parse_args(argv)
    try:
        data = json.loads(_read_input(args.input))
        if not isinstance(data, Mapping):
            raise ValueError("request JSON must be an object")
        payload = data.get("payload", {})
        if not isinstance(payload, Mapping):
            raise ValueError("payload must be an object")
        receipt = HybridSearchEvidence().evaluate(
            HybridSearchEvidenceRequest(
                subject_id=data.get("subject_id", ""),
                payload=dict(payload),
                budget=data.get("budget", 4.0),
                not_after=data.get("not_after"),
            )
        )
    except Exception as exc:
        print(json.dumps({"decision": "REFUSE", "reasons": [f"cli_input_error:{type(exc).__name__}:{exc}"]}, sort_keys=True))
        return 2
    print(json.dumps(receipt.as_dict(), indent=2, sort_keys=True))
    return 0 if receipt.decision is Decision.ALLOW else 2
