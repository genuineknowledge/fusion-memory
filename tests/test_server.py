from __future__ import annotations

import json
import threading
import unittest
from urllib import error, request

from fusion_memory import MemoryService
from fusion_memory.product import runtime_status_payload
from fusion_memory.server import serve


class ServerTests(unittest.TestCase):
    def test_runtime_status_defaults_to_sqlite_backend(self) -> None:
        status = runtime_status_payload()

        self.assertEqual(status["database"]["backend"], "sqlite")

    def test_status_endpoint_reports_readiness(self) -> None:
        ready = threading.Event()
        holder = {}

        def run_server() -> None:
            service = MemoryService()
            server = serve(service, host="127.0.0.1", port=0)
            holder["service"] = service
            holder["server"] = server
            ready.set()
            try:
                server.serve_forever()
            finally:
                server.server_close()
                service.close()

        thread = threading.Thread(target=run_server, daemon=True)
        thread.start()
        self.assertTrue(ready.wait(timeout=5))
        server = holder["server"]
        try:
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            status = _post_or_get(f"{base_url}/status")
            self.assertTrue(status["ok"])
            self.assertEqual(status["service"], "running")
            self.assertTrue(status["database"]["ok"])
            self.assertEqual(status["database"]["backend"], "sqlite")
            self.assertTrue(status["models"]["ok"])
            self.assertIn("version", status)
        finally:
            server.shutdown()
            thread.join(timeout=2)

    def test_status_endpoint_reports_explicit_postgres_backend(self) -> None:
        ready = threading.Event()
        holder = {}

        def run_server() -> None:
            service = MemoryService(storage_backend="postgres", store=_CloseOnlyStore())
            server = serve(service, host="127.0.0.1", port=0)
            holder["service"] = service
            holder["server"] = server
            ready.set()
            try:
                server.serve_forever()
            finally:
                server.server_close()
                service.close()

        thread = threading.Thread(target=run_server, daemon=True)
        thread.start()
        self.assertTrue(ready.wait(timeout=5))
        server = holder["server"]
        try:
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            status = _post_or_get(f"{base_url}/status")
            self.assertEqual(status["database"]["backend"], "postgres")
        finally:
            server.shutdown()
            thread.join(timeout=2)

    def test_persistent_http_server_adds_and_searches_memory(self) -> None:
        ready = threading.Event()
        holder = {}

        def run_server() -> None:
            service = MemoryService()
            server = serve(service, host="127.0.0.1", port=0)
            holder["service"] = service
            holder["server"] = server
            ready.set()
            try:
                server.serve_forever()
            finally:
                server.server_close()
                service.close()

        thread = threading.Thread(target=run_server, daemon=True)
        thread.start()
        self.assertTrue(ready.wait(timeout=5))
        server = holder["server"]
        try:
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            health = _post_or_get(f"{base_url}/health")
            self.assertTrue(health["ok"])

            scope = {"workspace_id": "w", "user_id": "u", "agent_id": "a"}
            add = _post_or_get(
                f"{base_url}/add",
                {
                    "input": {"role": "user", "content": "I prefer Qdrant for Atlas retrieval."},
                    "scope": scope,
                },
            )
            self.assertTrue(add["accepted_fact_ids"])

            search = _post_or_get(
                f"{base_url}/search",
                {
                    "query": "What do I prefer for Atlas retrieval?",
                    "scope": scope,
                    "options": {"limit": 3},
                },
            )
            self.assertTrue(search["candidates"])

            clear = _post_or_get(
                f"{base_url}/clear",
                {
                    "scope": scope,
                    "allow_cross_session": True,
                },
            )
            self.assertTrue(clear["ok"])
            self.assertEqual(clear["operation"], "clear_scope")
            self.assertGreaterEqual(clear["deleted"]["evidence_spans"], 1)

            after_clear = _post_or_get(
                f"{base_url}/search",
                {
                    "query": "What do I prefer for Atlas retrieval?",
                    "scope": scope,
                    "options": {"limit": 3},
                },
            )
            self.assertFalse(after_clear["candidates"])

            delete_alias = _post_or_get(
                f"{base_url}/delete",
                {
                    "scope": scope,
                    "allow_cross_session": True,
                },
            )
            self.assertTrue(delete_alias["ok"])
        finally:
            server.shutdown()
            thread.join(timeout=2)

    def test_post_errors_are_sanitized_for_beginner_clients(self) -> None:
        ready = threading.Event()
        holder = {}

        def run_server() -> None:
            service = MemoryService()
            server = serve(service, host="127.0.0.1", port=0)
            holder["service"] = service
            holder["server"] = server
            ready.set()
            try:
                server.serve_forever()
            finally:
                server.server_close()
                service.close()

        thread = threading.Thread(target=run_server, daemon=True)
        thread.start()
        self.assertTrue(ready.wait(timeout=5))
        server = holder["server"]
        try:
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            req = request.Request(
                f"{base_url}/search",
                data=json.dumps({"query": "missing scope"}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                request.urlopen(req, timeout=5)
                self.fail("expected HTTPError")
            except error.HTTPError as exc:
                response = exc.fp.read().decode("utf-8")
                payload = json.loads(response)
            self.assertEqual(payload["error"], "request_failed")
            self.assertIn("fusion-memory doctor", payload["message"])
            self.assertNotIn("ValueError", json.dumps(payload))
            self.assertNotIn("scope is required", json.dumps(payload))
        finally:
            server.shutdown()
            thread.join(timeout=2)


def _post_or_get(url: str, payload: dict | None = None) -> dict:
    if payload is None:
        with request.urlopen(url, timeout=5) as response:
            return json.loads(response.read().decode("utf-8"))
    req = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


class _CloseOnlyStore:
    def close(self) -> None:
        pass


if __name__ == "__main__":
    unittest.main()
