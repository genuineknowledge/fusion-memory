from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fusion_memory import Scope  # noqa: E402
from fusion_memory.chronology_backfill import backfill_chronology_graph  # noqa: E402
from fusion_memory.core.runtime_config import memory_service_from_env  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill persisted chronology graph tables from existing events and evidence spans.")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--user-id", default="beam_user")
    parser.add_argument("--agent-id", default="fusion_memory")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--include-session", action="store_true", default=True)
    parser.add_argument("--db", default=os.getenv("FUSION_MEMORY_DB", "postgresql://fusion:fusion@127.0.0.1:55432/fusion_memory"))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    backend = "postgres" if str(args.db).startswith(("postgresql://", "postgres://")) else None
    service = memory_service_from_env(args.db, storage_backend=backend)
    scope = Scope(
        workspace_id=args.workspace,
        user_id=args.user_id,
        agent_id=args.agent_id,
        run_id=args.run_id or args.workspace,
        session_id=args.session_id,
    )
    try:
        report = backfill_chronology_graph(service.store, scope, include_session=args.include_session)
    finally:
        service.close()
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
