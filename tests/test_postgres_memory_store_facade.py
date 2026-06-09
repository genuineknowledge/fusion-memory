from __future__ import annotations

import unittest

from fusion_memory import MemoryService
from fusion_memory.storage.postgres_store import PostgresMemoryStore
from fusion_memory.storage.sqlite_store import SQLiteMemoryStore


SERVICE_STORE_METHODS = [
    "close",
    "find_duplicate_span",
    "insert_span",
    "list_facts",
    "insert_encoding_decision",
    "insert_fact",
    "insert_event",
    "insert_fact_relation",
    "save_trace",
    "insert_audit_event",
    "insert_utility_example",
    "get_trace",
    "get_span",
    "get_fact",
    "get_event",
    "list_fact_relations",
    "list_events",
    "list_audit_events",
    "list_spans",
    "list_background_tasks",
    "next_background_tasks",
    "update_background_task",
    "list_encoding_decisions",
    "list_entity_profiles",
    "list_current_views",
    "enqueue_background_task",
    "has_event_edge",
    "insert_event_edge",
    "superseded_fact_ids",
    "upsert_current_view",
    "upsert_entity_profile",
    "search_spans",
    "search_facts",
    "search_events",
    "search_entity_profiles",
    "search_entities",
    "upsert_entity",
    "get_event_edge",
    "list_utility_examples",
]


class PostgresMemoryStoreFacadeTests(unittest.TestCase):
    def test_postgres_memory_store_exposes_service_store_interface(self) -> None:
        missing = [name for name in SERVICE_STORE_METHODS if not hasattr(PostgresMemoryStore, name)]
        self.assertEqual(missing, [])

    def test_memory_service_selects_sqlite_by_default_and_postgres_when_requested(self) -> None:
        default = MemoryService()
        try:
            self.assertIsInstance(default.store, SQLiteMemoryStore)
        finally:
            default.close()

        fake = FakeConnection()
        memory = MemoryService(
            "postgresql://example/fusion",
            storage_backend="postgres",
            store_connect=lambda dsn: fake,
        )
        try:
            self.assertIsInstance(memory.store, PostgresMemoryStore)
            self.assertEqual(memory.store.dsn, "postgresql://example/fusion")
            self.assertIs(memory.store.connect(), fake)
        finally:
            memory.close()
        self.assertTrue(fake.closed)

    def test_memory_service_accepts_injected_store_and_rejects_unknown_backend(self) -> None:
        store = InjectedStore()
        memory = MemoryService(store=store)
        memory.close()
        self.assertTrue(store.closed)

        with self.assertRaises(ValueError):
            MemoryService(storage_backend="unknown")


class FakeConnection:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class InjectedStore:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


if __name__ == "__main__":
    unittest.main()
