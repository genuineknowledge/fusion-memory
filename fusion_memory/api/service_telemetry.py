from __future__ import annotations

from typing import Any

from fusion_memory.retrieval.engine import sanitize_product_model_call


def _source_coverage(items: list[Any]) -> float:
    if not items:
        return 0.0
    covered = 0
    for item in items:
        if isinstance(item, dict):
            source_span_ids = item.get("source_span_ids") or item.get("candidate", {}).get("source_span_ids") or []
        else:
            source_span_ids = getattr(item, "source_span_ids", [])
        covered += int(bool(source_span_ids))
    return covered / len(items)


def _sanitize_model_call(component: str, source: Any, call: dict[str, Any]) -> dict[str, Any]:
    model = call.get("model") or getattr(source, "model", None)
    model_version = getattr(source, "version", None) or model or source.__class__.__name__
    out: dict[str, Any] = {
        "component": component,
        "model_version": model_version,
    }
    if model:
        out["model"] = model
    prompt_version = call.get("prompt_version") or call.get("prompt")
    if isinstance(prompt_version, str):
        prompt_version = prompt_version.splitlines()[0]
        out["prompt_version"] = prompt_version
    latency_ms = call.get("latency_ms")
    if isinstance(latency_ms, int | float):
        out["latency_ms"] = latency_ms
    usage = call.get("usage")
    if isinstance(usage, dict):
        out["usage"] = usage
    cost = call.get("cost")
    if isinstance(cost, int | float):
        out["cost"] = cost
    for key in ("text_count", "doc_count"):
        if isinstance(call.get(key), int):
            out[key] = call[key]
    return out


def _model_call_summary(model_calls: list[dict[str, Any]]) -> dict[str, Any]:
    usage_totals: dict[str, float] = {}
    for call in model_calls:
        usage = call.get("usage")
        if not isinstance(usage, dict):
            continue
        for key, value in usage.items():
            if isinstance(value, int | float):
                usage_totals[key] = usage_totals.get(key, 0.0) + float(value)
    return {
        "count": len(model_calls),
        "model_versions": sorted({str(call.get("model_version")) for call in model_calls if call.get("model_version")}),
        "total_latency_ms": sum(float(call.get("latency_ms", 0.0)) for call in model_calls if isinstance(call.get("latency_ms"), int | float)),
        "usage": usage_totals,
    }


def _labeled_precision(items: list[dict[str, Any]], labels: dict[str, bool], *, positive: bool) -> float | None:
    known = 0
    correct = 0
    for item in items:
        candidate = item.get("candidate", {})
        keys = [item.get("decision_id"), candidate.get("local_id"), candidate.get("text")]
        label = next((labels[key] for key in keys if key in labels), None)
        if label is None:
            continue
        known += 1
        correct += int(label is positive)
    return correct / known if known else None


def _model_call_sources(service: Any) -> list[tuple[str, Any]]:
    sources: list[tuple[str, Any]] = [
        ("embedder", getattr(service.store, "embedder", None)),
        ("extractor", service.extractor),
        ("extractor_client", getattr(service.extractor, "client", None)),
        ("async_extractor", service.async_extractor),
        ("async_extractor_client", getattr(service.async_extractor, "client", None)),
        ("reranker", service.reranker),
        ("retrieval_engine", service.retrieval_engine),
        ("retrieval_planner", getattr(service.retrieval_engine, "planner", None)),
        ("retrieval_registry", getattr(service.retrieval_engine, "registry", None)),
        ("retrieval_reranker", getattr(service.retrieval_engine, "reranker", None)),
    ]
    out: list[tuple[str, Any]] = []
    seen: set[int] = set()
    for component, source in sources:
        if source is None or id(source) in seen:
            continue
        seen.add(id(source))
        out.append((component, source))
    return out


def _model_call_marks(service: Any) -> dict[int, int]:
    marks: dict[int, int] = {}
    for _, source in _model_call_sources(service):
        calls = getattr(source, "calls", None)
        if isinstance(calls, list):
            marks[id(source)] = len(calls)
    return marks


def _model_calls_since(service: Any, marks: dict[int, int]) -> list[dict[str, Any]]:
    calls_out: list[dict[str, Any]] = []
    for component, source in _model_call_sources(service):
        calls = getattr(source, "calls", None)
        if not isinstance(calls, list):
            continue
        start = marks.get(id(source), 0)
        for call in calls[start:]:
            if isinstance(call, dict):
                calls_out.append(_sanitize_model_call(component, source, call))
            else:
                calls_out.append({"component": component, "model_version": getattr(source, "version", source.__class__.__name__)})
    return calls_out


def _product_model_calls_since(service: Any, marks: dict[int, int]) -> list[dict[str, Any]]:
    calls_out: list[dict[str, Any]] = []
    for component, source in _model_call_sources(service):
        calls = getattr(source, "calls", None)
        if not isinstance(calls, list):
            continue
        start = marks.get(id(source), 0)
        calls_out.extend(sanitize_product_model_call(component, source, call) for call in calls[start:])
    return calls_out
