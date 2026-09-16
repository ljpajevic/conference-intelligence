"""Per-request accounting for LLM calls: tokens, latency, throughput, cost.

Records are kept in memory, bounded, and reset on restart. Not a metrics
backend: a single instance scaling to zero is answered well enough by a
rolling window.

Rates are USD per million tokens, overridable via env because provider
pricing changes and a hardcoded number goes stale silently.
"""
import os
import time
from collections import deque
from dataclasses import dataclass, asdict

# Groq, openai/gpt-oss-120b, as published
INPUT_USD_PER_MTOK = float(os.getenv("LLM_INPUT_USD_PER_MTOK", "0.15"))
OUTPUT_USD_PER_MTOK = float(os.getenv("LLM_OUTPUT_USD_PER_MTOK", "0.60"))

MAX_RECORDS = 200


@dataclass
class CallRecord:
    source: str              # which code path made the call
    input_tokens: int
    output_tokens: int
    latency_s: float         # wall clock = what the caller waited
    generation_s: float      # provider-reported generation time, if given
    output_tokens_per_s: float
    cost_usd: float
    ok: bool


_records: deque[CallRecord] = deque(maxlen=MAX_RECORDS)


def _tokens(response) -> tuple[int, int]:
    """Token counts from a LangChain response.

    usage_metadata is the normalised field, token_usage is the provider's raw
    block. Try both, because which one is populated depends on the version.
    """
    normalised = getattr(response, "usage_metadata", None) or {}
    if normalised:
        return (int(normalised.get("input_tokens", 0)),
                int(normalised.get("output_tokens", 0)))

    meta = getattr(response, "response_metadata", None) or {}
    tu = meta.get("token_usage") or {}
    return int(tu.get("prompt_tokens", 0)), int(tu.get("completion_tokens", 0))


def _generation_seconds(response) -> float:
    """Provider-reported generation time. Groq returns completion_time."""
    meta = getattr(response, "response_metadata", None) or {}
    tu = meta.get("token_usage") or {}
    for key in ("completion_time", "total_time"):
        if tu.get(key):
            return float(tu[key])
    return 0.0


def cost_usd(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens * INPUT_USD_PER_MTOK
            + output_tokens * OUTPUT_USD_PER_MTOK) / 1_000_000


def record(response, *, source: str, latency_s: float, ok: bool = True) -> CallRecord:
    """Record one completed LLM call and return what was measured."""
    in_tok, out_tok = _tokens(response) if response is not None else (0, 0)
    gen_s = _generation_seconds(response) if response is not None else 0.0

    # prefer the provider's generation time for throughput: wall clock includes
    # queueing and network, which are not the model's rate
    basis = gen_s or latency_s
    tps = (out_tok / basis) if (basis > 0 and out_tok) else 0.0

    rec = CallRecord(
        source=source,
        input_tokens=in_tok,
        output_tokens=out_tok,
        latency_s=round(latency_s, 3),
        generation_s=round(gen_s, 3),
        output_tokens_per_s=round(tps, 1),
        cost_usd=round(cost_usd(in_tok, out_tok), 6),
        ok=ok,
    )
    _records.append(rec)
    return rec


class timed:
    """Context manager measuring wall-clock latency around an LLM call.

        with timed() as t:
            response = llm.invoke(...)
        usage.record(response, source="rationale", latency_s=t.elapsed)
    """
    def __enter__(self):
        self._start = time.perf_counter()
        self.elapsed = 0.0
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self._start


def summary() -> dict:
    """Aggregates over the rolling window, plus the most recent calls."""
    recs = list(_records)
    ok = [r for r in recs if r.ok]
    n = len(ok)

    # keys are always present, zeroed when nothing succeeded: a caller should
    # not have to handle two response shapes
    lat = sorted(r.latency_s for r in ok) or [0.0]
    tps = [r.output_tokens_per_s for r in ok if r.output_tokens_per_s]
    denom = n or 1

    return {
        "calls": len(recs),
        "failed": len(recs) - n,
        "window": MAX_RECORDS,
        "tokens": {
            "input_total": sum(r.input_tokens for r in ok),
            "output_total": sum(r.output_tokens for r in ok),
            "input_mean": round(sum(r.input_tokens for r in ok) / denom, 1),
            "output_mean": round(sum(r.output_tokens for r in ok) / denom, 1),
        },
        "latency_s": {
            "mean": round(sum(lat) / len(lat), 3),
            "p50": lat[len(lat) // 2],
            "p95": lat[min(int(len(lat) * 0.95), len(lat) - 1)],
            "max": lat[-1],
        },
        "output_tokens_per_s_mean": round(sum(tps) / len(tps), 1) if tps else 0.0,
        "cost_usd": {
            "total": round(sum(r.cost_usd for r in ok), 6),
            "per_request_mean": round(sum(r.cost_usd for r in ok) / denom, 6),
        },
        "rates_usd_per_mtok": {"input": INPUT_USD_PER_MTOK,
                               "output": OUTPUT_USD_PER_MTOK},
        "recent": [asdict(r) for r in recs[-10:]],
    }


def reset() -> None:
    _records.clear()
