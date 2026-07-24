import time
from collections import deque
from threading import Event, Thread

from langchain_core.messages import AIMessageChunk
from langchain_core.tools import StructuredTool

from agent import WorkspaceAgent
from contracts import (
    ModelCallMetricsEvent,
    ToolCallEvent,
    ToolResultEvent,
    TurnCancelledEvent,
)
from tool_execution import (
    CancellationToken,
    ToolExecutionMiddleware,
    ToolExecutionPolicy,
)


class ScriptedModel:
    def __init__(self, responses, *, tools_enabled=False):
        self.responses = (
            responses
            if isinstance(responses, deque)
            else deque(responses)
        )
        self.tools_enabled = tools_enabled

    def bind_tools(self, tools):
        return ScriptedModel(
            self.responses,
            tools_enabled=True,
        )

    def stream(self, messages):
        yield from self.responses.popleft()


def tool_call_response(tool_name, tool_call_id):
    return [
        AIMessageChunk(
            content="",
            tool_call_chunks=[
                {
                    "name": tool_name,
                    "args": "{}",
                    "id": tool_call_id,
                    "index": 0,
                }
            ],
        )
    ]


def test_read_only_timeout_returns_safe_error_and_model_continues():
    def slow_read():
        time.sleep(0.08)
        return "late-private-result"

    slow_tool = StructuredTool.from_function(
        func=slow_read,
        name="slow_read",
        description="Slow deterministic read.",
    )
    model = ScriptedModel(
        [
            tool_call_response("slow_read", "slow-call"),
            [AIMessageChunk(content="超时后安全收尾")],
        ]
    )
    middleware = ToolExecutionMiddleware(
        {
            "slow_read": ToolExecutionPolicy(
                risk="read_only",
                timeout_seconds=0.01,
                abandon_on_cancel=True,
            )
        },
        poll_interval_seconds=0.002,
    )
    agent = WorkspaceAgent(
        model=model,
        tools=[slow_tool],
        tool_execution_middleware=middleware,
    )

    events = list(agent.stream_turn("执行慢读取"))
    result = next(
        event
        for event in events
        if isinstance(event, ToolResultEvent)
    )

    assert result.status == "error"
    assert result.detail == "工具执行超时"
    assert result.error_type == "ToolExecutionTimeout"
    assert "late-private-result" not in repr(events)
    assert agent.messages[-1].content == "超时后安全收尾"


def test_agent_cancellation_stops_waiting_and_rolls_back_turn():
    started = Event()
    release = Event()

    def blocking_read():
        started.set()
        release.wait(timeout=2)
        return "uncommitted-result"

    blocking_tool = StructuredTool.from_function(
        func=blocking_read,
        name="blocking_read",
        description="Cancellable deterministic read.",
    )
    model = ScriptedModel(
        [
            tool_call_response("blocking_read", "blocking-call"),
        ]
    )
    agent = WorkspaceAgent(
        model=model,
        tools=[blocking_tool],
        default_tool_timeout_seconds=None,
    )
    original_messages = agent.messages
    stream = agent.stream_turn("开始可取消读取")

    first = next(stream)
    second = next(stream)
    assert isinstance(first, ModelCallMetricsEvent)
    assert isinstance(second, ToolCallEvent)

    collected = []
    failure = []

    def consume():
        try:
            collected.extend(list(stream))
        except BaseException as error:
            failure.append(error)

    consumer = Thread(target=consume)
    consumer.start()
    assert started.wait(timeout=1)
    assert agent.cancel_active_turn("client_disconnect") is True
    consumer.join(timeout=1)
    release.set()

    assert not consumer.is_alive()
    assert failure == []
    assert any(
        isinstance(event, ToolResultEvent)
        and event.status == "skipped"
        and event.error_type == "ToolExecutionCancelled"
        for event in collected
    )
    assert collected[-1] == TurnCancelledEvent(
        reason="client_disconnect"
    )
    assert agent.messages is original_messages
    assert agent.cancel_active_turn() is False


def test_side_effect_policy_finishes_started_atomic_boundary():
    started = Event()
    release = Event()
    token = CancellationToken()
    middleware = ToolExecutionMiddleware(
        {
            "write_like": ToolExecutionPolicy(
                risk="workspace_write",
                timeout_seconds=None,
                abandon_on_cancel=False,
            )
        }
    )
    outcome = []

    def action():
        started.set()
        release.wait(timeout=1)
        return "committed"

    worker = Thread(
        target=lambda: outcome.append(
            middleware.execute("write_like", action, token)
        )
    )
    worker.start()
    assert started.wait(timeout=1)
    assert token.cancel("user") is True
    release.set()
    worker.join(timeout=1)

    assert outcome == ["committed"]
    assert not worker.is_alive()
