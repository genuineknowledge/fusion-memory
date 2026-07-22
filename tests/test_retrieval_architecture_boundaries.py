import ast
from pathlib import Path


PRODUCTION_ROOTS = [
    Path("fusion_memory/api"),
    Path("fusion_memory/retrieval"),
    Path("fusion_memory/mcp_runtime.py"),
]
PRODUCT_PRODUCTION_MODULES = [
    Path("fusion_memory/retrieval/product_engine.py"),
    Path("fusion_memory/retrieval/product_evidence_pack.py"),
    Path("fusion_memory/retrieval/product_planner.py"),
    *Path("fusion_memory/retrieval/providers").glob("product_*.py"),
]


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
