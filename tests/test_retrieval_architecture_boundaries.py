import ast
import re
from pathlib import Path


PRODUCTION_ROOTS = [
    Path("fusion_memory/api"),
    Path("fusion_memory/retrieval"),
    Path("fusion_memory/mcp_runtime.py"),
]
PRODUCT_PRODUCTION_MODULES = [
    Path("fusion_memory/retrieval/product_engine.py"),
    Path("fusion_memory/retrieval/evidence_pack.py"),
    Path("fusion_memory/retrieval/query_planner.py"),
    Path("fusion_memory/retrieval/providers/base.py"),
    Path("fusion_memory/retrieval/providers/registry.py"),
]
LEGACY_RETRIEVAL_MODULES = (
    Path("fusion_memory/api/service_helpers.py"),
    Path("fusion_memory/retrieval/candidate_provider.py"),
    Path("fusion_memory/retrieval/pack_contract.py"),
    Path("fusion_memory/retrieval/pipeline.py"),
    Path("fusion_memory/retrieval/preservation.py"),
    Path("fusion_memory/retrieval/providers/raw.py"),
    Path("fusion_memory/retrieval/providers/structured.py"),
    Path("fusion_memory/retrieval/raw_evidence_quota.py"),
    Path("fusion_memory/retrieval/retrieval_trace.py"),
    Path("fusion_memory/retrieval/scoring.py"),
)
TRANSITIONAL_PRODUCT_MODULES = (
    Path("fusion_memory/retrieval/product_evidence_pack.py"),
    Path("fusion_memory/retrieval/product_planner.py"),
    Path("fusion_memory/retrieval/providers/product_base.py"),
    Path("fusion_memory/retrieval/providers/product_registry.py"),
)


def _production_python() -> str:
    files = []
    for root in PRODUCTION_ROOTS:
        if root.is_file():
            files.append(root)
        else:
            files.extend(root.rglob("*.py"))
    return "\n".join(path.read_text(encoding="utf-8") for path in files)


def test_production_retrieval_has_no_beam_categories_or_benchmark_mode() -> None:
    source = _production_python()
    for forbidden in (
        "query_type_hint",
        'mode == "benchmark"',
        'mode in {"balanced", "benchmark"}',
        '"contradiction_resolution"',
        '"multi_session_reasoning"',
        '"preference_following"',
        '"instruction_following"',
        '"information_extraction"',
    ):
        assert forbidden not in source


def test_product_providers_do_not_import_or_reference_memory_service() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in Path("fusion_memory/retrieval/providers").rglob("*.py")
    )
    assert "fusion_memory.api.service" not in source
    assert "MemoryService" not in source
    assert "context.service" not in source


def test_product_retrieval_uses_only_canonical_module_paths() -> None:
    from fusion_memory.retrieval.evidence_pack import ProductEvidencePackBuilder
    from fusion_memory.retrieval.providers.base import CandidateProvider, ProviderContext
    from fusion_memory.retrieval.providers.registry import ProductProviderRegistry
    from fusion_memory.retrieval.query_planner import ProductQueryPlanner

    assert ProductQueryPlanner
    assert ProductEvidencePackBuilder
    assert CandidateProvider
    assert ProviderContext
    assert ProductProviderRegistry
    assert all(not path.exists() for path in TRANSITIONAL_PRODUCT_MODULES)


def test_legacy_retrieval_modules_types_and_category_config_are_removed() -> None:
    assert all(not path.exists() for path in LEGACY_RETRIEVAL_MODULES)

    models_source = Path("fusion_memory/core/models.py").read_text(encoding="utf-8")
    config_source = Path("fusion_memory/core/config.py").read_text(encoding="utf-8")
    production_source = _production_python()
    assert "class QueryPlan:" not in models_source
    assert "raw_evidence_quotas" not in config_source
    assert "DEFAULT_RAW_EVIDENCE_QUOTAS" not in config_source
    for forbidden in (
        r"\bQueryPlan\b",
        r"\bRetrievalExecutionContext\b",
        r"\bRecallContext\b",
        r"context\.service",
        r"service\._(?:topic|aggregation|event_ordering|preserve|apply_quality)",
    ):
        assert re.search(forbidden, production_source) is None


def test_production_imports_do_not_reference_legacy_or_transitional_modules() -> None:
    imported_modules: set[str] = set()
    for root in PRODUCTION_ROOTS:
        paths = [root] if root.is_file() else root.rglob("*.py")
        for path in paths:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    imported_modules.add(node.module)
                elif isinstance(node, ast.Import):
                    imported_modules.update(alias.name for alias in node.names)

    forbidden_modules = {
        ".".join(path.with_suffix("").parts)
        for path in (*LEGACY_RETRIEVAL_MODULES, *TRANSITIONAL_PRODUCT_MODULES)
    }
    assert imported_modules.isdisjoint(forbidden_modules)


def test_product_production_modules_do_not_import_legacy_scenario_helpers() -> None:
    imported_modules: set[str] = set()
    for path in PRODUCT_PRODUCTION_MODULES:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module)
            elif isinstance(node, ast.Import):
                imported_modules.update(alias.name for alias in node.names)

    assert not {
        module
        for module in imported_modules
        if module.startswith("fusion_memory.retrieval.aggregation_")
        or "scenario" in module
    }


def test_memory_service_is_within_facade_size_budget() -> None:
    lines = Path("fusion_memory/api/service.py").read_text(encoding="utf-8").splitlines()
    assert len(lines) <= 1200
