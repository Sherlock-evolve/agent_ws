import json
from collections import deque

import pytest
from langchain_core.messages import AIMessageChunk

import session_store
import tools as workspace_tools
from agent import WorkspaceAgent
from contracts import (
    ApprovalDecision,
    ApprovalRequiredEvent,
    ApprovalResolvedEvent,
    SessionSavedEvent,
    ToolResultEvent,
)
from persistent_session import (
    PersistentSession,
    PersistentSessionOpenError,
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


def write_call_response(tool_call_id, path, content):
    return [
        AIMessageChunk(
            content="",
            tool_call_chunks=[
                {
                    "name": "write_file",
                    "args": json.dumps(
                        {
                            "path": path,
                            "content": content,
                        },
                        ensure_ascii=False,
                    ),
                    "id": tool_call_id,
                    "index": 0,
                }
            ],
        )
    ]


def make_agent(model):
    return WorkspaceAgent(
        model=model,
        tools=[workspace_tools.write_file],
        approval_required_tools={"write_file"},
        approval_preparers={
            "write_file": workspace_tools.prepare_write_file,
        },
    )


def use_isolated_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(
        workspace_tools,
        "WORKSPACE_ROOT",
        tmp_path.resolve(),
    )
    monkeypatch.setattr(
        session_store,
        "SESSION_STORE_ROOT",
        tmp_path / ".agent_sessions",
    )


def stop_at_approval(session):
    stream = session.stream_turn("创建可恢复文件")
    while True:
        envelope = next(stream)
        if isinstance(envelope.event, ApprovalRequiredEvent):
            stream.close()
            return envelope.event


def test_pending_write_survives_restart_and_resumes_exact_turn(
    tmp_path,
    monkeypatch,
):
    use_isolated_paths(tmp_path, monkeypatch)
    responses = deque(
        [
            write_call_response(
                "recoverable-write",
                "result.txt",
                "恢复后的内容",
            ),
            [AIMessageChunk(content="恢复审批后完成回答。")],
        ]
    )

    first_session = PersistentSession.open(
        "recoverable",
        lambda: make_agent(ScriptedModel(responses)),
    )
    approval = stop_at_approval(first_session)

    assert approval.preview
    assert not (tmp_path / "result.txt").exists()
    assert first_session.has_pending_approval
    assert session_store.list_sessions() == ["recoverable"]
    pending_path = (
        session_store.SESSION_STORE_ROOT
        / "recoverable.pending.json"
    )
    assert pending_path.exists()

    restored = PersistentSession.open(
        "recoverable",
        lambda: make_agent(ScriptedModel(responses)),
    )
    summary = restored.pending_approval_event()
    assert restored.has_pending_approval
    assert summary.tool_name == "write_file"
    assert summary.args["content"] == "<6 characters>"

    resumed = restored.stream_resume_pending_approval()
    events = []
    decision = None
    while True:
        try:
            envelope = resumed.send(decision)
        except StopIteration:
            break
        decision = None
        events.append(envelope.event)
        if isinstance(envelope.event, ApprovalRequiredEvent):
            assert "result.txt" in envelope.event.preview
            decision = ApprovalDecision(
                tool_call_id=envelope.event.tool_call_id,
                approved=True,
            )

    assert any(
        isinstance(event, ApprovalResolvedEvent)
        and event.outcome == "approved"
        for event in events
    )
    assert any(
        isinstance(event, ToolResultEvent)
        and event.status == "success"
        for event in events
    )
    assert isinstance(events[-1], SessionSavedEvent)
    assert (tmp_path / "result.txt").read_text(
        encoding="utf-8"
    ) == "恢复后的内容"
    assert not pending_path.exists()
    assert not restored.has_pending_approval
    assert restored.agent.messages[-1].content == "恢复审批后完成回答。"

    reopened = PersistentSession.open(
        "recoverable",
        lambda: make_agent(ScriptedModel([])),
    )
    assert not reopened.has_pending_approval
    assert reopened.agent.messages[-1].content == "恢复审批后完成回答。"


def test_pending_approval_rejects_changed_committed_snapshot(
    tmp_path,
    monkeypatch,
):
    use_isolated_paths(tmp_path, monkeypatch)
    responses = deque(
        [
            write_call_response(
                "stale-write",
                "stale.txt",
                "must-not-write",
            )
        ]
    )
    session = PersistentSession.open(
        "stale",
        lambda: make_agent(ScriptedModel(responses)),
    )
    stop_at_approval(session)

    changed_snapshot = session.agent.export_snapshot()
    changed_snapshot["memory_summary"] = "外部更新后的摘要"
    session_store.save("stale", changed_snapshot)

    with pytest.raises(
        PersistentSessionOpenError,
        match="待审批轮次语义无效",
    ):
        PersistentSession.open(
            "stale",
            lambda: make_agent(ScriptedModel([])),
        )
    assert not (tmp_path / "stale.txt").exists()
