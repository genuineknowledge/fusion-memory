from __future__ import annotations

import math
from typing import Any, Protocol

from fusion_memory.core.models import EvidencePack
from fusion_memory.core.text import stable_hash
from fusion_memory.retrieval.context import (
    ProductQueryPlan,
    ProviderKind,
    RetrievalContext,
    RetrievalResult,
    SearchRequest,
)
from fusion_memory.retrieval.tracing import (
    sanitize_dimension,
    sanitize_query_intent_telemetry,
)


class RetrievalUnavailable(RuntimeError):
    pass


_TRACE_STAGES = ("plan", "recall", "fusion", "selection")
_PROVIDER_KINDS = frozenset({"vector", "lexical", "temporal", "entity", "chronology"})
_MODEL_CALL_COMPONENTS = frozenset(
    {
        "embedder",
        "extractor",
        "extractor_client",
        "async_extractor",
        "async_extractor_client",
        "reranker",
        "retrieval_engine",
        "retrieval_planner",
        "retrieval_registry",
        "retrieval_reranker",
    }
)
_MODEL_CALL_NUMERIC_FIELDS = ("latency_ms", "cost", "text_count", "doc_count")
_MODEL_CALL_USAGE_FIELDS = (
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "cached_tokens",
    "completion_tokens",
    "input_tokens",
    "output_tokens",
    "prompt_tokens",
    "reasoning_tokens",
    "total_tokens",
)


def _safe_numeric(value: object) -> float | None:
    if type(value) not in (int, float):
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    return max(0.0, numeric)


def _safe_dimension(value: object, *, readable: frozenset[str] = frozenset()) -> str | None:
    if type(value) not in (str, int, float, bool):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    label = str(value)
    if label in readable:
        return label
    return f"hashed_{stable_hash(label)[:16]}"


def sanitize_product_model_call(
    component: str,
    source: object,
    call: object,
) -> dict[str, Any]:
    """Project model telemetry into the confidentiality-safe product contract."""
    call_data = call if isinstance(call, dict) else {}
    projected: dict[str, Any] = {
        "component": _safe_dimension(component, readable=_MODEL_CALL_COMPONENTS)
    }

    model_value = call_data.get("model")
    if model_value is None:
        model_value = getattr(source, "model", None)
    model = _safe_dimension(model_value)
    if model is not None:
        projected["model"] = model

    model_version_value = call_data.get("model_version")
    if model_version_value is None:
        model_version_value = getattr(source, "version", None)
    if model_version_value is None:
        model_version_value = (
            model_value if model_value is not None else source.__class__.__name__
        )
    model_version = _safe_dimension(model_version_value)
    if model_version is not None:
        projected["model_version"] = model_version

    for field in _MODEL_CALL_NUMERIC_FIELDS:
        numeric = _safe_numeric(call_data.get(field))
        if numeric is not None:
            projected[field] = numeric

    usage_data = call_data.get("usage")
    if isinstance(usage_data, dict):
        usage = {
            field: numeric
            for field in _MODEL_CALL_USAGE_FIELDS
            if (numeric := _safe_numeric(usage_data.get(field))) is not None
        }
        if usage:
            projected["usage"] = usage
    return projected


def summarize_product_model_calls(model_calls: list[dict[str, Any]]) -> dict[str, Any]:
    usage_totals: dict[str, float] = {}
    for call in model_calls:
        usage = call.get("usage")
        if not isinstance(usage, dict):
            continue
        for field in _MODEL_CALL_USAGE_FIELDS:
            numeric = _safe_numeric(usage.get(field))
            if numeric is not None:
                usage_totals[field] = usage_totals.get(field, 0.0) + numeric
    return {
        "count": len(model_calls),
        "model_versions": sorted(
            {
                model_version
                for call in model_calls
                if isinstance((model_version := call.get("model_version")), str)
            }
        ),
        "total_latency_ms": sum(
            numeric
            for call in model_calls
            if (numeric := _safe_numeric(call.get("latency_ms"))) is not None
        ),
        "usage": usage_totals,
    }


def sanitize_retrieval_trace(trace: dict[str, Any]) -> dict[str, Any]:
    """Keep only the product trace contract, which contains no raw memory text."""
    sanitized: dict[str, Any] = {}
    stages = trace.get("stages")
    if isinstance(stages, list | tuple):
        sanitized["stages"] = [stage for stage in stages if stage in _TRACE_STAGES]
    mode = trace.get("mode")
    if mode in {"fast", "balanced"}:
        sanitized["mode"] = mode
    if "intent" in trace:
        sanitized["intent"] = sanitize_dimension(trace["intent"])

    providers: list[dict[str, Any]] = []
    provider_rows = trace.get("providers")
    if isinstance(provider_rows, list | tuple):
        for row in provider_rows:
            if not isinstance(row, dict):
                continue
            kind = row.get("kind")
            provider: dict[str, Any] = {
                "kind": (
                    kind
                    if isinstance(kind, str) and kind in _PROVIDER_KINDS
                    else sanitize_dimension(kind)
                )
            }
            count = row.get("count")
            if isinstance(count, int) and not isinstance(count, bool):
                provider["count"] = max(0, count)
            elapsed_ms = row.get("elapsed_ms")
            if isinstance(elapsed_ms, int | float) and not isinstance(elapsed_ms, bool):
                provider["elapsed_ms"] = max(0.0, float(elapsed_ms))
            failure_code = row.get("failure_code")
            provider["failure_code"] = (
                None if failure_code is None else sanitize_dimension(failure_code)
            )
            providers.append(provider)
    if providers:
        sanitized["providers"] = providers

    filtered_count = trace.get("filtered_count")
    if isinstance(filtered_count, int) and not isinstance(filtered_count, bool):
        sanitized["filtered_count"] = max(0, filtered_count)
    selected_ids = trace.get("selected_ids")
    if isinstance(selected_ids, list | tuple):
        sanitized["selected_ids"] = [
            value
            for value in selected_ids
            if isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
        ]
    durations = trace.get("stage_durations_ms")
    if isinstance(durations, dict):
        sanitized["stage_durations_ms"] = {
            stage: max(0.0, float(durations[stage]))
            for stage in _TRACE_STAGES
            if isinstance(durations.get(stage), int | float)
            and not isinstance(durations.get(stage), bool)
        }
    if trace.get("reranker_failure") is not None:
        sanitized["reranker_failure"] = sanitize_dimension(trace["reranker_failure"])
    if trace.get("planner_fallback") is not None:
        fallback = trace["planner_fallback"]
        sanitized["planner_fallback"] = (
            fallback if fallback == "invalid_plan" else sanitize_dimension(fallback)
        )
    query_intent_telemetry = sanitize_query_intent_telemetry(
        trace.get("query_intent_telemetry")
    )
    if query_intent_telemetry:
        sanitized["query_intent_telemetry"] = query_intent_telemetry
    return sanitized


def prepare_retrieval_engine_options(
    options: dict[str, Any] | None,
    config: Any,
) -> dict[str, Any]:
    options = dict(options or {})
    supported_options = {
        "allow_cross_session",
        "deadline",
        "enabled_providers",
        "enabled_sources",
        "include_trace",
        "limit",
        "mode",
        "time_range",
        "token_budget",
    }
    if any(option not in supported_options for option in options):
        raise ValueError("unsupported retrieval options")

    mode = options.get("mode", "fast")
    if type(mode) is not str or mode not in {"fast", "balanced"}:
        raise ValueError("mode must be fast or balanced")

    source_provider_kinds = {
        "raw": {
            ProviderKind.VECTOR,
            ProviderKind.LEXICAL,
            ProviderKind.TEMPORAL,
            ProviderKind.CHRONOLOGY,
        },
        "exact": {ProviderKind.LEXICAL},
        "entities": {ProviderKind.ENTITY},
        "facts": {ProviderKind.VECTOR, ProviderKind.LEXICAL},
        "events": {
            ProviderKind.VECTOR,
            ProviderKind.TEMPORAL,
            ProviderKind.CHRONOLOGY,
        },
        "views": {ProviderKind.LEXICAL},
        "profiles": {ProviderKind.LEXICAL, ProviderKind.ENTITY},
    }
    enabled_sources_option = options.get("enabled_sources")
    if enabled_sources_option is None:
        source_values: list[str] | None = None
    elif type(enabled_sources_option) is str:
        source_values = [enabled_sources_option]
    elif isinstance(enabled_sources_option, (list, tuple, set, frozenset)):
        source_values = list(enabled_sources_option)
    else:
        source_values = None
    if source_values is None and enabled_sources_option is not None:
        raise ValueError("enabled_sources contains an unsupported source family")
    if source_values is not None and any(
        type(value) is not str or value not in source_provider_kinds
        for value in source_values
    ):
        raise ValueError("enabled_sources contains an unsupported source family")

    enabled_providers_option = options.get("enabled_providers")
    if enabled_providers_option is not None:
        if isinstance(enabled_providers_option, ProviderKind) or type(enabled_providers_option) is str:
            provider_values = [enabled_providers_option]
        elif isinstance(enabled_providers_option, (list, tuple, set, frozenset)):
            provider_values = list(enabled_providers_option)
        else:
            provider_values = None
        if provider_values is None:
            raise ValueError("enabled_providers contains an unsupported provider")
        provider_by_name = {provider.value: provider for provider in ProviderKind}
        parsed_providers: set[ProviderKind] = set()
        for value in provider_values:
            if isinstance(value, ProviderKind):
                parsed_providers.add(value)
                continue
            if type(value) is not str:
                raise ValueError("enabled_providers contains an unsupported provider")
            provider = provider_by_name.get(value)
            if provider is None:
                raise ValueError("enabled_providers contains an unsupported provider")
            parsed_providers.add(provider)
        enabled_providers = frozenset(parsed_providers)
    elif source_values is None:
        enabled_providers = None
    else:
        enabled_providers = frozenset(
            provider
            for source_name in source_values
            for provider in source_provider_kinds[source_name]
        )

    limit = options.get("limit", config.retrieval_output_n)
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ValueError("limit must be positive")
    token_budget = options.get("token_budget")
    if token_budget is not None and (
        not isinstance(token_budget, int)
        or isinstance(token_budget, bool)
        or token_budget < 0
    ):
        raise ValueError("token_budget must be a non-negative integer")

    return {
        "allow_cross_session": bool(options.get("allow_cross_session", False)),
        "deadline": options.get("deadline"),
        "enabled_providers": enabled_providers,
        "enabled_sources": tuple(source_values) if source_values is not None else None,
        "include_trace": bool(options.get("include_trace", True)),
        "limit": int(limit),
        "mode": mode,
        "time_range": options.get("time_range"),
        "token_budget": int(token_budget) if token_budget is not None else None,
    }


class RetrievalEngine(Protocol):
    def search(
        self,
        context: RetrievalContext,
        request: SearchRequest,
        plan: ProductQueryPlan | None = None,
    ) -> RetrievalResult: ...

    def search_with_plan(
        self,
        context: RetrievalContext,
        request: SearchRequest,
        plan: ProductQueryPlan,
    ) -> RetrievalResult: ...

    def build_evidence_pack(
        self,
        context: RetrievalContext,
        request: SearchRequest,
        result: RetrievalResult,
        token_budget: int,
    ) -> EvidencePack: ...


def build_product_retrieval_engine(
    repository: Any,
    config: Any,
    reranker: Any,
    planner: Any | None = None,
    *,
    query_intent_refiner: Any | None = None,
    query_intent_refiner_min_confidence: float = 0.70,
    query_intent_refiner_mode: str = "off",
) -> RetrievalEngine:
    from fusion_memory.retrieval.product_engine import ProductRetrievalEngine
    from fusion_memory.retrieval.evidence_pack import ProductEvidencePackBuilder
    from fusion_memory.retrieval.providers.chronology import ChronologyProvider
    from fusion_memory.retrieval.providers.entity import EntityProvider
    from fusion_memory.retrieval.providers.lexical import LexicalProvider
    from fusion_memory.retrieval.providers.registry import ProductProviderRegistry
    from fusion_memory.retrieval.providers.temporal import TemporalProvider
    from fusion_memory.retrieval.providers.vector import VectorProvider
    from fusion_memory.retrieval.query_planner import ProductQueryPlanner

    pack_builder = ProductEvidencePackBuilder(repository, config)
    registry = ProductProviderRegistry(
        [
            VectorProvider(repository),
            LexicalProvider(repository),
            TemporalProvider(repository),
            EntityProvider(repository),
            ChronologyProvider(repository),
        ]
    )
    product_planner = (
        planner
        if planner is not None
        else ProductQueryPlanner(
            intent_refiner=query_intent_refiner,
            intent_refiner_min_confidence=query_intent_refiner_min_confidence,
            intent_refiner_mode=query_intent_refiner_mode,
        )
    )
    return ProductRetrievalEngine(
        product_planner,
        registry,
        pack_builder=pack_builder,
        reranker=reranker,
        mmr_lambda=config.mmr_lambda,
    )
