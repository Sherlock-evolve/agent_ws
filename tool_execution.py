"""Policy-driven synchronous tool execution with timeout and cancellation.

The middleware owns the execution boundary only. Conversation budgets,
duplicate-call detection and human approval remain Agent concerns.
"""

from __future__ import annotations

import math
import queue
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from threading import Event, Lock, Thread
from typing import Literal


ToolRisk = Literal[
    "read_only",
    "workspace_write",
    "external_side_effect",
]
CANCELLATION_REASONS = frozenset(
    {"user", "client_disconnect", "shutdown"}
)


class ToolExecutionError(RuntimeError):
    """Base class for safe execution-control failures."""


class ToolExecutionTimeout(ToolExecutionError):
    """A cancellable tool exceeded its configured execution deadline."""


class ToolExecutionCancelled(ToolExecutionError):
    """The caller cancelled the active turn."""


class CancellationToken:
    """A thread-safe, one-way cancellation signal."""

    def __init__(self) -> None:
        self._event = Event()
        self._reason = "user"
        self._lock = Lock()

    def cancel(self, reason: str = "user") -> bool:
        if reason not in CANCELLATION_REASONS:
            raise ValueError("cancellation reason is not supported")
        with self._lock:
            if self._event.is_set():
                return False
            self._reason = reason
            self._event.set()
            return True

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str:
        with self._lock:
            return self._reason

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise ToolExecutionCancelled("The active turn was cancelled.")


@dataclass(frozen=True)
class ToolExecutionPolicy:
    """Execution policy attached to one registered tool."""

    risk: ToolRisk = "read_only"
    timeout_seconds: float | None = 30.0
    abandon_on_cancel: bool = True

    def __post_init__(self) -> None:
        if self.risk not in {
            "read_only",
            "workspace_write",
            "external_side_effect",
        }:
            raise ValueError("invalid tool risk")
        timeout = self.timeout_seconds
        if timeout is not None and (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or timeout <= 0
        ):
            raise ValueError("timeout_seconds must be a positive finite number")
        if type(self.abandon_on_cancel) is not bool:
            raise ValueError("abandon_on_cancel must be a boolean")
        if self.risk != "read_only" and self.abandon_on_cancel:
            raise ValueError(
                "side-effecting tools cannot be abandoned after execution starts"
            )
        if self.risk != "read_only" and self.timeout_seconds is not None:
            raise ValueError(
                "side-effecting tools require a cooperative timeout "
                "and must use timeout_seconds=None"
            )


class ToolExecutionMiddleware:
    """Execute registered actions under immutable per-tool policies."""

    def __init__(
        self,
        policies: Mapping[str, ToolExecutionPolicy] | None = None,
        *,
        default_policy: ToolExecutionPolicy | None = None,
        poll_interval_seconds: float = 0.02,
        monotonic_clock: Callable[[], float] | None = None,
    ) -> None:
        configured_policies = dict(policies or {})
        if any(
            not isinstance(name, str)
            or not name
            or not isinstance(policy, ToolExecutionPolicy)
            for name, policy in configured_policies.items()
        ):
            raise ValueError(
                "tool execution policies require names and ToolExecutionPolicy values"
            )
        if default_policy is None:
            default_policy = ToolExecutionPolicy()
        if not isinstance(default_policy, ToolExecutionPolicy):
            raise TypeError("default_policy must be ToolExecutionPolicy")
        if (
            isinstance(poll_interval_seconds, bool)
            or not isinstance(poll_interval_seconds, (int, float))
            or not math.isfinite(float(poll_interval_seconds))
            or poll_interval_seconds <= 0
        ):
            raise ValueError(
                "poll_interval_seconds must be a positive finite number"
            )
        if monotonic_clock is not None and not callable(monotonic_clock):
            raise TypeError("monotonic_clock must be callable")

        self._policies = configured_policies
        self.default_policy = default_policy
        self.poll_interval_seconds = float(poll_interval_seconds)
        self.monotonic_clock = monotonic_clock or time.monotonic

    def policy_for(self, tool_name: str) -> ToolExecutionPolicy:
        if not isinstance(tool_name, str) or not tool_name:
            raise ValueError("tool_name must be a non-empty string")
        return self._policies.get(tool_name, self.default_policy)

    def execute(
        self,
        tool_name: str,
        action: Callable[[], object],
        cancellation_token: CancellationToken,
    ) -> object:
        """Run one action.

        Read-only actions run in a daemon worker so the caller can stop waiting
        after a timeout or cancellation. Side-effecting actions run inline:
        cancellation is checked before they start, and their atomic boundary is
        allowed to finish once entered.
        """

        if not callable(action):
            raise TypeError("tool action must be callable")
        if not isinstance(cancellation_token, CancellationToken):
            raise TypeError("cancellation_token must be CancellationToken")

        policy = self.policy_for(tool_name)
        cancellation_token.raise_if_cancelled()

        if policy.risk != "read_only":
            return action()

        return self._execute_cancellable(
            action,
            cancellation_token,
            policy,
        )

    def _execute_cancellable(
        self,
        action: Callable[[], object],
        cancellation_token: CancellationToken,
        policy: ToolExecutionPolicy,
    ) -> object:
        outcomes: queue.Queue[tuple[bool, object]] = queue.Queue(maxsize=1)

        def run_action() -> None:
            try:
                outcomes.put((True, action()))
            except BaseException as error:
                outcomes.put((False, error))

        worker = Thread(
            target=run_action,
            name="workspace-agent-tool",
            daemon=True,
        )
        worker.start()

        started_at = self.monotonic_clock()
        deadline = (
            started_at + float(policy.timeout_seconds)
            if policy.timeout_seconds is not None
            else None
        )
        while True:
            if cancellation_token.cancelled and policy.abandon_on_cancel:
                raise ToolExecutionCancelled("The active turn was cancelled.")

            wait_seconds = self.poll_interval_seconds
            if deadline is not None:
                remaining = deadline - self.monotonic_clock()
                if remaining <= 0:
                    raise ToolExecutionTimeout(
                        "The tool exceeded its execution deadline."
                    )
                wait_seconds = min(wait_seconds, remaining)

            try:
                succeeded, value = outcomes.get(timeout=wait_seconds)
            except queue.Empty:
                continue

            if succeeded:
                return value
            if isinstance(value, BaseException):
                raise value
            raise ToolExecutionError("The tool returned an invalid outcome.")
