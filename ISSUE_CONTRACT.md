# Issue contract — Hybrid Search Claim Fence

## Problem
Hybrid BM25/vector answers over-claim support from weak hits.

## Desired outcome
A bounded, open, testable implementation of **Hybrid Search Claim Fence** that demonstrates Score support strength and fence claims below threshold as non-factual suggestions.

## Non-goals
- Weaviate affiliation or proprietary integration
- Portfolio-wide scale/performance claims
- UI marketing site

## Acceptance
1. Mechanism module implements allow + refuse with structured receipts
2. pytest behavioral suite green
3. operate.py cold-start produces JSON receipt
4. Non-affiliation disclaimer preserved
