from __future__ import annotations

import concurrent.futures
import threading


class BlockingUserService:
    def __init__(self) -> None:
        self.active_users: set[str] = set()
        self.first_started = threading.Event()
        self.second_started = threading.Event()
        self.release = threading.Event()

    def run_for_user(self, user_id: str) -> None:
        self.active_users.add(user_id)
        (self.first_started if user_id == "user-a" else self.second_started).set()
        self.release.wait(timeout=1)
        self.active_users.remove(user_id)


def test_different_users_can_execute_in_parallel() -> None:
    service = BlockingUserService()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(service.run_for_user, "user-a")
        service.first_started.wait(timeout=1)
        second = executor.submit(service.run_for_user, "user-b")
        service.second_started.wait(timeout=1)
        assert service.active_users == {"user-a", "user-b"}
        service.release.set()
        first.result(timeout=1)
        second.result(timeout=1)
