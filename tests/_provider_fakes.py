from __future__ import annotations

import copy
from typing import Callable


class OutcomeSequence:
    """Record provider requests and return or raise queued outcomes."""

    def __init__(
        self,
        outcomes,
        *,
        snapshot: Callable[[dict], dict] = copy.deepcopy,
    ) -> None:
        self._outcomes = list(outcomes)
        self._snapshot = snapshot
        self.calls: list[dict] = []

    def take(self, request: dict):
        self.calls.append(self._snapshot(request))
        if not self._outcomes:
            raise AssertionError("unexpected extra API call")
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome
