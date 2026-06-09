from __future__ import annotations

import unittest
from datetime import datetime, timezone
from typing import Any

from fusion_memory import AuthorizationError, MemoryService, Scope


def ts(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


class RecordingAuthorizer:
    def __init__(self, denied: set[str] | None = None) -> None:
        self.denied = denied or set()
        self.calls: list[tuple[str, Scope, dict[str, Any]]] = []

    def authorize(self, operation: str, scope: Scope, context: dict[str, Any] | None = None) -> None:
        self.calls.append((operation, scope, dict(context or {})))
        if operation in self.denied:
            raise AuthorizationError(f"denied: {operation}")


class AuthorizerTests(unittest.TestCase):
    def test_default_authorizer_preserves_existing_add_search_behavior(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a")
        memory.add("I prefer Qdrant for Atlas retrieval.", scope, ts("2026-06-01T10:00:00+00:00"))

        result = memory.search("Qdrant Atlas", scope)

        self.assertTrue(any("Qdrant" in candidate.text for candidate in result.candidates))

    def test_deny_add_stops_persistence_and_audit(self) -> None:
        authorizer = RecordingAuthorizer({"memory.add"})
        memory = MemoryService(authorizer=authorizer)
        scope = Scope(workspace_id="w", user_id="u", agent_id="a")

        with self.assertRaises(AuthorizationError):
            memory.add("I prefer Qdrant for Atlas retrieval.", scope, ts("2026-06-01T10:00:00+00:00"))

        self.assertEqual(memory.store.list_spans(scope), [])
        self.assertEqual(memory.store.list_audit_events(scope), [])
        self.assertEqual(authorizer.calls[0][0], "memory.add")

    def test_deny_search_happens_before_trace_and_audit_write(self) -> None:
        authorizer = RecordingAuthorizer({"memory.search"})
        memory = MemoryService(authorizer=authorizer)
        scope = Scope(workspace_id="w", user_id="u", agent_id="a")
        memory.add("I prefer Qdrant for Atlas retrieval.", scope, ts("2026-06-01T10:00:00+00:00"))

        with self.assertRaises(AuthorizationError):
            memory.search("Qdrant Atlas", scope)

        self.assertEqual(memory.store.list_audit_events(scope, event_type="memory.search"), [])
        self.assertEqual(authorizer.calls[-1][0], "memory.search")

    def test_deny_read_helpers_and_reports(self) -> None:
        authorizer = RecordingAuthorizer(
            {
                "memory.get",
                "memory.debug_trace",
                "memory.history",
                "memory.events.compare",
                "memory.timeline",
                "memory.report.encoding",
                "memory.report.profiles",
                "memory.summary.read",
                "memory.summary.refresh",
                "memory.tasks.read",
                "memory.tasks.process",
            }
        )
        memory = MemoryService(authorizer=authorizer)
        scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")
        result = memory.add("I tested BM25 yesterday.", scope, ts("2026-06-03T12:00:00+00:00"))
        memory.add("I added dense retrieval today.", scope, ts("2026-06-05T12:00:00+00:00"))
        events = memory.store.list_events(scope)

        with self.assertRaises(AuthorizationError):
            memory.get(result.span_ids[0], "span", scope)
        with self.assertRaises(AuthorizationError):
            memory.debug_trace(result.trace_id, scope)
        with self.assertRaises(AuthorizationError):
            memory.history(scope)
        with self.assertRaises(AuthorizationError):
            memory.compare_events(events[0].event_id, events[1].event_id, scope)
        with self.assertRaises(AuthorizationError):
            memory.timeline(None, scope)
        with self.assertRaises(AuthorizationError):
            memory.encoding_report(scope)
        with self.assertRaises(AuthorizationError):
            memory.profile_report(scope)
        with self.assertRaises(AuthorizationError):
            memory.get_session_summaries(scope)
        with self.assertRaises(AuthorizationError):
            memory.refresh_session_summary(scope)
        with self.assertRaises(AuthorizationError):
            memory.list_background_tasks(scope)
        with self.assertRaises(AuthorizationError):
            memory.process_background_tasks(scope)

        denied_calls = [operation for operation, _, _ in authorizer.calls if operation in authorizer.denied]
        self.assertEqual(
            denied_calls,
            [
                "memory.get",
                "memory.debug_trace",
                "memory.history",
                "memory.events.compare",
                "memory.timeline",
                "memory.report.encoding",
                "memory.report.profiles",
                "memory.summary.read",
                "memory.summary.refresh",
                "memory.tasks.read",
                "memory.tasks.process",
            ],
        )

    def test_authorizer_receives_operation_scope_and_context(self) -> None:
        authorizer = RecordingAuthorizer()
        memory = MemoryService(authorizer=authorizer)
        scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")
        memory.add("I prefer Qdrant for Atlas retrieval.", scope, ts("2026-06-01T10:00:00+00:00"))
        memory.search("Qdrant Atlas", scope, options={"allow_cross_session": True, "limit": 3})

        operations = [operation for operation, _, _ in authorizer.calls]
        self.assertIn("memory.add", operations)
        self.assertIn("memory.search", operations)

        search_call = next(call for call in authorizer.calls if call[0] == "memory.search")
        self.assertEqual(search_call[1], scope)
        self.assertTrue(search_call[2]["allow_cross_session"])
        self.assertFalse(search_call[2]["include_session"])
        self.assertEqual(search_call[2]["limit"], 3)


if __name__ == "__main__":
    unittest.main()
