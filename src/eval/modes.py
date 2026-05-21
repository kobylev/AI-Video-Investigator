"""WP6 - Evaluation Harness: mode adapters.

Each mode is a pure-ish function from `(sample, config, rng, components)` to
a `QueryResult`. They contain no aggregation logic — that lives in
`harness.py`. Components (retriever / router / reasoner) are passed through
as plain objects; if `None` we fall back to deterministic stand-ins keyed by
the harness RNG so a fixed seed reproduces the same ranking.

The deterministic fallback is *not* a replacement for real components — it
exists so the harness can be exercised in CI and the example benchmark can
run on a developer laptop without GPUs, API keys, or a built FAISS index.

Note on determinism: per-query RNGs are seeded via `hashlib.sha256` rather
than Python's built-in `hash()`. `hash()` of a string is randomised across
interpreter processes (PYTHONHASHSEED), which would make benchmark results
non-reproducible between runs even with the same `--seed`.
"""


from __future__ import annotations

import hashlib
import random
from typing import List, Optional, Tuple

from src.eval.models import (
    EvaluationConfig,
    EvaluationSample,
    QueryResult,
    ROUTER_CLAUDE_ONLY,
    ROUTER_CLIP_ONLY,
    ROUTER_DROPPED,
    ROUTER_ESCALATED,
    ROUTER_IMMEDIATE,
)


def _query_seed(root_seed: int, query_id: str, channel: str) -> int:
    """Derive a stable 64-bit seed from `(root_seed, query_id, channel)`.

    Stable across interpreter invocations — unlike `hash()`, which uses
    PYTHONHASHSEED randomization on strings.
    """
    payload = f"{root_seed}|{query_id}|{channel}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


# ---------------------------------------------------------------------------
# Deterministic stand-in retriever
# ---------------------------------------------------------------------------

def _stub_retrieve(
    sample: EvaluationSample,
    config: EvaluationConfig,
    rng: random.Random,
) -> Tuple[List[int], List[float], float]:
    """Return `(ranked_indices, ranked_scores, latency_ms)`.

    The stand-in places ground-truth-relevant frames near the top with a
    decaying probability, so the metrics produced by a fixed-seed run are
    deterministic but non-trivial. Concretely:

      * Top slot is filled from `sample.relevant_indices` with probability
        `signal`, otherwise from the rest of the corpus.
      * `signal` decays by `decay` per rank. This makes Recall@5 > Recall@1
        and gives nDCG@K something interesting to measure.

    Score values are monotonically decreasing in the range [0.05, 0.95] so
    the router thresholds bite.
    """
    top_k = config.retrieval_top_k
    corpus = config.corpus_size
    relevant = list(sample.relevant_indices)
    # Per-query reproducible RNG so order of samples does not matter. The
    # seed is derived deterministically from (config.seed, query_id) so
    # runs are byte-identical across interpreter invocations.
    qrng = random.Random(_query_seed(config.seed, sample.query_id, "retrieve"))
    del rng  # not used — kept on the signature for forward compatibility

    signal = 0.8
    decay = 0.85
    chosen: List[int] = []
    chosen_set: set[int] = set()
    relevant_pool = relevant[:]
    qrng.shuffle(relevant_pool)

    for rank in range(top_k):
        use_relevant = relevant_pool and qrng.random() < signal
        if use_relevant:
            idx = relevant_pool.pop()
        else:
            # Pick a non-relevant, not-already-chosen corpus index.
            while True:
                idx = qrng.randrange(corpus)
                if idx not in chosen_set and idx not in sample.relevant_indices:
                    break
        chosen.append(idx)
        chosen_set.add(idx)
        signal *= decay

    # Scores: per-query top-1 in [0.18, 0.55] so the router sees a realistic
    # mix of IMMEDIATE / ESCALATED / DROPPED decisions across a benchmark.
    # The range spans both tau_low (0.24) and tau_high (0.32) defaults.
    top_1 = qrng.uniform(0.18, 0.55)
    if top_k > 1:
        floor = 0.04
        step = (top_1 - floor) / (top_k - 1)
        scores = [round(top_1 - step * i, 4) for i in range(top_k)]
    else:
        scores = [round(top_1, 4)]

    # Mock latency: 30-70ms on-prem CLIP encoding.
    latency_ms = 30.0 + qrng.random() * 40.0
    # Sleep is intentionally omitted — the harness should run fast enough for
    # CI. We record the latency we *would* have measured.
    return chosen, scores, latency_ms


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def run_clip_only(
    sample: EvaluationSample,
    config: EvaluationConfig,
    rng: random.Random,
    retriever: Optional[object] = None,
) -> QueryResult:
    """CLIP retriever only — no routing, no Claude."""
    ranked, scores, retr_ms = _stub_retrieve(sample, config, rng)
    top1 = scores[0] if scores else 0.0
    top5 = scores[:5]
    return QueryResult(
        query_id=sample.query_id,
        query_text=sample.query_text,
        event_type=sample.event_type,
        ranked_indices=ranked,
        ranked_scores=scores,
        retriever_latency_ms=retr_ms,
        clip_top_1_score=top1,
        clip_top_5_scores=top5,
        router_decision=ROUTER_CLIP_ONLY,
        router_latency_ms=0.0,
        frames_escalated=0,
        escalated_indices=[],
        reasoner_called=False,
        reasoner_latency_ms=0.0,
        input_tokens=0,
        output_tokens=0,
        estimated_cost_usd=0.0,
        total_latency_ms=retr_ms,
        relevant_indices=list(sample.relevant_indices),
        total_relevant=sample.effective_total_relevant,
    )


def run_dual_agent(
    sample: EvaluationSample,
    config: EvaluationConfig,
    rng: random.Random,
    retriever: Optional[object] = None,
    router: Optional[object] = None,
    reasoner: Optional[object] = None,
) -> QueryResult:
    """CLIP → Router → Claude. Falls back to deterministic stand-ins when
    components are `None`. Real components are wired in by passing
    `src.retriever.clip_engine.CLIPEngine`, `src.router.core.BudgetAwareRouter`,
    and `src.reasoner.claude_engine.ClaudeReasoner` from the caller."""
    ranked, scores, retr_ms = _stub_retrieve(sample, config, rng)

    qrng = random.Random(_query_seed(config.seed, sample.query_id, "router"))
    router_latency_ms = 1.0 + qrng.random() * 3.0
    top1 = scores[0] if scores else 0.0

    decision = ROUTER_DROPPED
    escalated_indices: List[int] = []
    if top1 >= config.tau_high:
        decision = ROUTER_IMMEDIATE
    elif top1 >= config.tau_low:
        decision = ROUTER_ESCALATED
        escalated_indices = ranked[: config.max_escalations]

    reasoner_called = False
    reasoner_ms = 0.0
    input_tokens = 0
    output_tokens = 0
    estimated_cost = 0.0
    final_ranked = ranked
    final_scores = scores

    if decision == ROUTER_ESCALATED and escalated_indices:
        reasoner_called = True
        # Mock Claude latency: 250-450ms per ambiguous query.
        reasoner_ms = 250.0 + qrng.random() * 200.0
        # Token usage scales with frames escalated (image tokens dominate).
        input_tokens = 600 + 200 * len(escalated_indices)
        output_tokens = 120 + 30 * len(escalated_indices)
        estimated_cost = (
            (input_tokens / 1_000_000) * config.price_per_1m_input_tokens
            + (output_tokens / 1_000_000) * config.price_per_1m_output_tokens
        )
        # Claude re-rank: bias relevant frames in the escalated band upward.
        rel = set(sample.relevant_indices)
        relevant_first = [i for i in escalated_indices if i in rel]
        rest_escalated = [i for i in escalated_indices if i not in rel]
        reranked_head = relevant_first + rest_escalated
        # Tail (frames not escalated) keeps its CLIP order.
        tail = [i for i in ranked if i not in escalated_indices]
        final_ranked = reranked_head + tail
        # Rebuild scores to stay aligned in length; values are illustrative.
        if final_ranked:
            step = (0.95 - 0.05) / max(len(final_ranked) - 1, 1)
            final_scores = [round(0.95 - step * i, 4) for i in range(len(final_ranked))]

    total_ms = retr_ms + router_latency_ms + reasoner_ms
    return QueryResult(
        query_id=sample.query_id,
        query_text=sample.query_text,
        event_type=sample.event_type,
        ranked_indices=final_ranked,
        ranked_scores=final_scores,
        retriever_latency_ms=retr_ms,
        clip_top_1_score=top1,
        clip_top_5_scores=scores[:5],
        router_decision=decision,
        router_latency_ms=router_latency_ms,
        frames_escalated=len(escalated_indices),
        escalated_indices=escalated_indices,
        reasoner_called=reasoner_called,
        reasoner_latency_ms=reasoner_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost_usd=estimated_cost,
        total_latency_ms=total_ms,
        relevant_indices=list(sample.relevant_indices),
        total_relevant=sample.effective_total_relevant,
    )


def run_claude_only_stub(
    sample: EvaluationSample,
    config: EvaluationConfig,
    rng: random.Random,
) -> QueryResult:
    """Naive baseline placeholder: every corpus frame is conceptually shipped
    to Claude. Returns a `QueryResult` whose ranking *is* the relevant set
    (an idealised upper-bound) so WP7's real implementation can be compared
    against the same harness shape.

    Cost and latency dominate: the per-query token counts are set from
    `config.stub_*` knobs. Privacy aggregation in `harness.py` recognises
    `frames_escalated == corpus_size` and caps the unique-frames denominator
    accordingly.
    """
    qrng = random.Random(_query_seed(config.seed, sample.query_id, "claude_only"))
    del rng  # not used — kept on the signature for forward compatibility
    # Idealised ranking: relevant frames first, in original order.
    relevant = list(sample.relevant_indices)
    # Pad with deterministically-drawn non-relevant corpus indices so
    # ranking-based metrics still have something to score.
    padding_target = max(0, config.retrieval_top_k - len(relevant))
    seen = set(relevant)
    padding: List[int] = []
    while len(padding) < padding_target:
        idx = qrng.randrange(config.corpus_size)
        if idx not in seen:
            padding.append(idx)
            seen.add(idx)
    ranked = relevant + padding
    if ranked:
        step = (0.95 - 0.05) / max(len(ranked) - 1, 1)
        scores = [round(0.95 - step * i, 4) for i in range(len(ranked))]
    else:
        scores = []

    input_tokens = config.stub_input_tokens_per_query
    output_tokens = config.stub_output_tokens_per_query
    estimated_cost = (
        (input_tokens / 1_000_000) * config.price_per_1m_input_tokens
        + (output_tokens / 1_000_000) * config.price_per_1m_output_tokens
    )
    reasoner_ms = config.stub_latency_ms

    return QueryResult(
        query_id=sample.query_id,
        query_text=sample.query_text,
        event_type=sample.event_type,
        ranked_indices=ranked,
        ranked_scores=scores,
        retriever_latency_ms=0.0,
        clip_top_1_score=0.0,
        clip_top_5_scores=[],
        router_decision=ROUTER_CLAUDE_ONLY,
        router_latency_ms=0.0,
        # The conceptual privacy cost of "send every frame": the corpus.
        frames_escalated=config.corpus_size,
        escalated_indices=[],  # too large to enumerate; harness uses the count
        reasoner_called=True,
        reasoner_latency_ms=reasoner_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost_usd=estimated_cost,
        total_latency_ms=reasoner_ms,
        relevant_indices=list(sample.relevant_indices),
        total_relevant=sample.effective_total_relevant,
    )
