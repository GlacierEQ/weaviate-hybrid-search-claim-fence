# Hybrid Search Evidence Claim Fence

Independent GlacierEQ specialist component aligned to hybrid-search evidence verification themes.

> **Not affiliated.** This repository is not affiliated with, endorsed by, employed by, or deployed at Weaviate. No proprietary access, production deployment, customer impact, provider authentication, or company partnership is claimed.

## Problem

Hybrid sparse/vector retrieval can produce plausible-looking ranked results without enough source-diverse evidence to support a factual claim. A retrieval score is not itself proof.

## Implemented mechanism

`src/hybrid_search_evidence.py` is the canonical mechanism. A provider adapter supplies normalized BM25/vector scores and immutable evidence identities. The verifier:

- combines normalized sparse/vector scores using explicit `alpha`;
- validates bounded inputs, score ranges, identities, digests, request expiry, and work budget;
- ranks evidence deterministically;
- requires configurable supporting-hit, source-diversity, support-weight, and contradiction limits;
- binds decisions to an index snapshot and deterministic evidence digest;
- supports expected-evidence rebinding so changed evidence refuses rather than silently passing;
- returns an explicit ALLOW/REFUSE receipt.

`scripts/verify_claim.py` exercises baseline allowance, exact rebinding, contradiction refusal, and evidence-tampering refusal. The full pytest suite is the deterministic/adversarial verification surface.

## Reproduce

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python scripts/verify_claim.py
```

## Role and family boundary

This repository is a **specialist hybrid-search evidence verifier**. It owns deterministic fusion-policy evaluation and evidence binding. It does not own provider retrieval, vector indexing, authentication, deployment, or live Weaviate integration. Those require separate adapters and authenticated runtime receipts.

Its distinct value is the combination of normalized hybrid scoring with claim-evidence sufficiency, source diversity, contradiction accounting, bounded work, snapshot identity, and deterministic rebinding. That mechanism is consequential and separate from ACL/entitlement claim fencing, which is an authorization-specific concern.

## Truth boundary

- No live Weaviate connection or provider attestation is claimed.
- No customer, revenue, latency, scale, retrieval-quality, or production-use claim is made without separate evidence.
- Passing local/CI verification proves only the checked-in deterministic verifier contract at the exact tested source revision.
- Future source changes invalidate historical working proof until a new exact-head receipt passes.
