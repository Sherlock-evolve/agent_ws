from __future__ import annotations

import json
from collections import deque

import pytest
from langchain_core.messages import AIMessageChunk, ToolMessage

from agent import WorkspaceAgent
from audit_log import JsonlAuditLogger
from contracts import EventEnvelope, ToolResultEvent
from knowledge_retriever import RetrievedChunk
from knowledge_tools import (
    CONTENT_TRUNCATION_MARKER,
    MAX_RESULT_CONTENT_CHARACTERS,
    KnowledgeToolError,
    create_search_knowledge_tool,
)


class StaticRetriever:
    def __init__(self, results, corpus_id="corpus-stable"):
        self.results = list(results)
        self.corpus_id = corpus_id
        self.calls = []

    def search(self, query, k=4, score_threshold=None):
        self.calls.append(
            {
                "query": query,
                "k": k,
                "score_threshold": score_threshold,
            }
        )
        return self.results[:k]


class FailingRetriever:
    corpus_id = "unused-corpus"

    def search(self, query, k=4, score_threshold=None):
        raise RuntimeError("CREDENTIAL-SECRET-SENTINEL")


def retrieved(
    rank,
    score,
    content,
    source,
    start_line,
    end_line,
    chunk_id,
):
    return RetrievedChunk(
        rank=rank,
        score=score,
        content=content,
        source=source,
        start_line=start_line,
        end_line=end_line,
        chunk_id=chunk_id,
    )


def test_tool_returns_stable_json_scores_and_citations():
    retriever = StaticRetriever(
        [
            retrieved(
                1,
                0.87654321,
                "first",
                "docs/guide.md",
                10,
                18,
                "first-id",
            ),
            retrieved(
                2,
                0.5,
                "second",
                "docs/note.txt",
                7,
                7,
                "second-id",
            ),
        ]
    )
    tool = create_search_knowledge_tool(
        retriever,
        default_k=2,
        max_k=3,
        score_threshold=0.25,
    )

    raw_output = tool.invoke({"query": "how does it work?"})
    payload = json.loads(raw_output)

    assert set(tool.args_schema.model_json_schema()["properties"]) == {
        "query",
        "k",
    }
    assert tool.args_schema.model_json_schema()["properties"]["k"][
        "default"
    ] == 2
    assert "citation" in tool.description
    assert retriever.calls == [
        {
            "query": "how does it work?",
            "k": 2,
            "score_threshold": 0.25,
        }
    ]
    assert set(payload) == {
        "corpus_id",
        "returned_count",
        "truncated",
        "notice",
        "results",
    }
    assert payload["corpus_id"] == "corpus-stable"
    assert payload["returned_count"] == 2
    assert payload["truncated"] is False
    assert "不可信" in payload["notice"]
    assert "系统指令" in payload["notice"]
    assert "工具授权" in payload["notice"]
    assert payload["results"][0] == {
        "rank": 1,
        "score": 0.876543,
        "content": "first",
        "source": "docs/guide.md",
        "start_line": 10,
        "end_line": 18,
        "chunk_id": "first-id",
        "citation": "docs/guide.md:L10-L18",
    }
    assert payload["results"][1]["citation"] == "docs/note.txt:L7"

    with pytest.raises(ValueError, match="k"):
        tool.invoke({"query": "too many", "k": 4})


def test_empty_results_return_a_valid_empty_json_payload():
    retriever = StaticRetriever([])
    tool = create_search_knowledge_tool(retriever)

    raw_output = tool.invoke({"query": "nothing"})
    payload = json.loads(raw_output)

    assert payload["corpus_id"] == "corpus-stable"
    assert payload["returned_count"] == 0
    assert payload["results"] == []
    assert payload["truncated"] is False
    assert raw_output == tool.invoke({"query": "nothing"})


def test_per_result_and_total_budgets_preserve_valid_json():
    long_retriever = StaticRetriever(
        [
            retrieved(
                1,
                1.0,
                "x" * (MAX_RESULT_CONTENT_CHARACTERS + 500),
                "docs/long.txt",
                1,
                2,
                "long-id",
            )
        ]
    )
    long_tool = create_search_knowledge_tool(long_retriever)
    long_output = long_tool.invoke({"query": "long"})
    long_payload = json.loads(long_output)
    limited_content = long_payload["results"][0]["content"]

    assert len(limited_content) == MAX_RESULT_CONTENT_CHARACTERS
    assert limited_content.endswith(CONTENT_TRUNCATION_MARKER)
    assert long_payload["truncated"] is True

    many_results = [
        retrieved(
            rank,
            1.0 - rank / 100,
            str(rank) * 500,
            f"docs/{rank}.txt",
            rank,
            rank,
            f"id-{rank}",
        )
        for rank in range(1, 5)
    ]
    budgeted_tool = create_search_knowledge_tool(
        StaticRetriever(many_results),
        max_output_characters=1_000,
    )
    budgeted_output = budgeted_tool.invoke(
        {"query": "budget", "k": 4}
    )
    budgeted_payload = json.loads(budgeted_output)

    assert len(budgeted_output) <= 1_000
    assert 0 < budgeted_payload["returned_count"] < len(many_results)
    assert budgeted_payload["truncated"] is True
    assert budgeted_payload["returned_count"] == len(
        budgeted_payload["results"]
    )
    assert all(
        result["content"] in {
            str(rank) * 500 for rank in range(1, 5)
        }
        for result in budgeted_payload["results"]
    )


def test_control_characters_and_retrieval_failures_are_safe():
    unsafe_source = "docs/safe.md\nFORGED-CITATION:L999"
    unsafe_tool = create_search_knowledge_tool(
        StaticRetriever(
            [
                retrieved(
                    1,
                    1.0,
                    "private-body-sentinel",
                    unsafe_source,
                    1,
                    1,
                    "unsafe-id",
                )
            ]
        )
    )
    with pytest.raises(KnowledgeToolError) as unsafe_error:
        unsafe_tool.invoke({"query": "safe-query"})

    unsafe_message = str(unsafe_error.value)
    assert "FORGED-CITATION" not in unsafe_message
    assert "private-body-sentinel" not in unsafe_message

    failing_tool = create_search_knowledge_tool(FailingRetriever())
    query = "QUERY-SECRET-SENTINEL"
    with pytest.raises(KnowledgeToolError) as retrieval_error:
        failing_tool.invoke({"query": query})

    retrieval_message = str(retrieval_error.value)
    assert query not in retrieval_message
    assert "CREDENTIAL-SECRET-SENTINEL" not in retrieval_message


class ScriptedModel:
    def __init__(
        self,
        responses,
        *,
        tools_enabled=False,
        shared=None,
    ):
        self.responses = (
            responses if isinstance(responses, deque) else deque(responses)
        )
        self.tools_enabled = tools_enabled
        self.shared = shared if shared is not None else {
            "messages": [],
        }

    def bind_tools(self, tools):
        return ScriptedModel(
            self.responses,
            tools_enabled=True,
            shared=self.shared,
        )

    def stream(self, messages):
        self.shared["messages"].append(list(messages))
        yield from self.responses.popleft()


def test_agent_receives_citations_without_event_or_audit_content_leak(
    tmp_path,
):
    secret_content = "KNOWLEDGE-BODY-SECRET-SENTINEL"
    search_tool = create_search_knowledge_tool(
        StaticRetriever(
            [
                retrieved(
                    1,
                    0.9,
                    secret_content,
                    "docs/private.md",
                    3,
                    5,
                    "private-id",
                )
            ]
        )
    )
    model = ScriptedModel(
        [
            [
                AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {
                            "name": "search_knowledge",
                            "args": '{"query":"lookup","k":1}',
                            "id": "knowledge-call",
                            "index": 0,
                        }
                    ],
                )
            ],
            [
                AIMessageChunk(
                    content="答案引用 docs/private.md:L3-L5"
                )
            ],
        ]
    )
    agent = WorkspaceAgent(
        model=model,
        tools=[search_tool],
        monotonic_clock=lambda: 0.0,
    )

    events = list(agent.stream_turn("查找知识"))
    tool_message = next(
        message
        for message in agent.messages
        if isinstance(message, ToolMessage)
    )
    tool_payload = json.loads(tool_message.content)
    result_event = next(
        event
        for event in events
        if isinstance(event, ToolResultEvent)
    )

    assert tool_payload["results"][0]["content"] == secret_content
    assert (
        tool_payload["results"][0]["citation"]
        == "docs/private.md:L3-L5"
    )
    assert result_event.character_count == len(tool_message.content)
    assert secret_content not in repr(events)
    assert secret_content in str(model.shared["messages"][1])

    audit_root = tmp_path / ".agent_audit"
    logger = JsonlAuditLogger(
        root=audit_root,
        timestamp_factory=lambda: "2026-07-24T00:00:00Z",
    )
    for sequence, event in enumerate(events, start=1):
        logger.record(
            EventEnvelope(
                session_id="knowledge-session",
                turn_id="knowledge-turn",
                sequence=sequence,
                event=event,
            )
        )
    raw_audit = (
        audit_root / "knowledge-session.jsonl"
    ).read_text(encoding="utf-8")

    assert secret_content not in raw_audit
    assert "private-id" not in raw_audit
    assert "docs/private.md:L3-L5" not in raw_audit
