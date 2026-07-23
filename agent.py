import json
from collections.abc import Generator, Iterator
from copy import deepcopy
from dataclasses import dataclass, field
from threading import Lock
from typing import Literal

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    message_chunk_to_message,
)


SYSTEM_PROMPT = (
    "你是一位人工智能老师，也是当前项目的工作区助手。"
    "需要了解项目文件时，请使用工具获取真实信息，不要猜测。"
    "请用通俗、准确的方式回答。"
)


@dataclass(frozen=True)
class TokenEvent:
    text: str


@dataclass(frozen=True)
class ToolCallEvent:
    tool_call_id: str
    step: int
    name: str
    args: dict


@dataclass(frozen=True)
class ToolResultEvent:
    tool_call_id: str
    name: str
    status: Literal["success", "error", "skipped"]
    character_count: int
    detail: str = ""


@dataclass(frozen=True)
class SystemEvent:
    message: str


AgentEvent = TokenEvent | ToolCallEvent | ToolResultEvent | SystemEvent


class _FrozenDict(dict):
    def _immutable(self, *args, **kwargs):
        raise TypeError("事件参数不可修改")

    __setitem__ = _immutable
    __delitem__ = _immutable
    __ior__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable


def _freeze(value):
    if isinstance(value, dict):
        return _FrozenDict({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return value


@dataclass
class _TurnState:
    tool_call_count: int = 0
    seen_tool_calls: set[tuple[str, str]] = field(default_factory=set)
    tool_budget_exhausted: bool = False


class WorkspaceAgent:
    """单会话 Agent；同一实例同一时间只允许运行一个事件流。"""

    def __init__(
        self,
        model,
        tools,
        max_agent_loops=5,
        max_tool_calls=8,
    ):
        self.model = model
        self.tools = list(tools)
        self.max_agent_loops = max_agent_loops
        self.max_tool_calls = max_tool_calls
        self.tools_by_name = {tool.name: tool for tool in self.tools}
        self.model_with_tools = self.model.bind_tools(self.tools)
        self.messages = [SystemMessage(content=SYSTEM_PROMPT)]
        self._turn_lock = Lock()

    def stream_turn(self, question: str) -> Iterator[AgentEvent]:
        """运行一轮并产生事件；提前停止消费时应关闭返回的生成器。"""
        if not self._turn_lock.acquire(blocking=False):
            raise RuntimeError("同一 WorkspaceAgent 不能同时运行多个对话轮次")

        try:
            yield from self._run_turn_transaction(question)
        finally:
            self._turn_lock.release()

    def _run_turn_transaction(self, question: str) -> Iterator[AgentEvent]:
        working_messages = list(self.messages)
        working_messages.append(HumanMessage(content=question))
        state = _TurnState()
        answered = False
        task_stopped = False

        for step in range(1, self.max_agent_loops + 1):
            tools_allowed = (
                step < self.max_agent_loops
                and not state.tool_budget_exhausted
            )
            active_model = self.model_with_tools if tools_allowed else self.model
            response = yield from self._stream_response(
                active_model,
                working_messages,
            )

            if response is None:
                yield SystemEvent(
                    message="模型未返回任何消息，当前任务已停止。"
                )
                task_stopped = True
                break

            working_messages.append(response)

            if not response.tool_calls:
                self.messages = working_messages
                answered = True
                break

            for tool_call in response.tool_calls:
                yield from self._execute_tool_call(
                    tool_call=tool_call,
                    step=step,
                    tools_allowed=tools_allowed,
                    state=state,
                    working_messages=working_messages,
                )

        if not answered and not task_stopped:
            yield SystemEvent(
                message=(
                    f"Agent 循环达到 {self.max_agent_loops} 次上限，"
                    "已停止。"
                )
            )

    def _stream_response(
        self,
        active_model,
        working_messages: list,
    ) -> Generator[AgentEvent, None, AIMessage | None]:
        response_chunk = None

        for chunk in active_model.stream(working_messages):
            if isinstance(chunk.content, str):
                text = chunk.content
            else:
                text = chunk.text

            if text:
                yield TokenEvent(text=text)

            if response_chunk is None:
                response_chunk = chunk
            else:
                response_chunk = response_chunk + chunk

        if response_chunk is None:
            return None

        return message_chunk_to_message(response_chunk)

    def _execute_tool_call(
        self,
        tool_call,
        step: int,
        tools_allowed: bool,
        state: _TurnState,
        working_messages: list,
    ) -> Iterator[AgentEvent]:
        tool_call_id = tool_call["id"]
        tool_name = tool_call["name"]
        internal_args = deepcopy(tool_call["args"])
        event_args = _freeze(deepcopy(internal_args))
        signature = (
            tool_name,
            json.dumps(
                internal_args,
                ensure_ascii=False,
                sort_keys=True,
            ),
        )
        yield ToolCallEvent(
            tool_call_id=tool_call_id,
            step=step,
            name=tool_name,
            args=event_args,
        )

        if not tools_allowed:
            detail = "当前轮次禁止调用工具"
            self._append_tool_message(
                messages=working_messages,
                content=(
                    "当前轮次不允许调用工具，本次调用未执行。"
                    "请根据已有信息直接回答。"
                ),
                tool_call_id=tool_call_id,
            )
            yield ToolResultEvent(
                tool_call_id=tool_call_id,
                name=tool_name,
                status="skipped",
                character_count=0,
                detail=detail,
            )
            return

        if signature in state.seen_tool_calls:
            detail = "重复调用"
            self._append_tool_message(
                messages=working_messages,
                content="重复工具调用已跳过，请使用之前相同工具调用的结果。",
                tool_call_id=tool_call_id,
            )
            yield ToolResultEvent(
                tool_call_id=tool_call_id,
                name=tool_name,
                status="skipped",
                character_count=0,
                detail=detail,
            )
            return

        state.seen_tool_calls.add(signature)

        if state.tool_call_count >= self.max_tool_calls:
            state.tool_budget_exhausted = True
            detail = "工具预算已耗尽"
            self._append_tool_message(
                messages=working_messages,
                content=(
                    "工具预算已耗尽，本次调用未执行。"
                    "请根据已有信息回答。"
                ),
                tool_call_id=tool_call_id,
            )
            yield ToolResultEvent(
                tool_call_id=tool_call_id,
                name=tool_name,
                status="skipped",
                character_count=0,
                detail=detail,
            )
            return

        state.tool_call_count += 1
        if state.tool_call_count >= self.max_tool_calls:
            state.tool_budget_exhausted = True

        try:
            selected_tool = self.tools_by_name[tool_name]
            tool_result_text = str(selected_tool.invoke(internal_args))
            tool_status = "success"
        except Exception as error:
            tool_result_text = f"工具执行失败：{error}"
            tool_status = "error"

        self._append_tool_message(
            messages=working_messages,
            content=tool_result_text,
            tool_call_id=tool_call_id,
        )
        yield ToolResultEvent(
            tool_call_id=tool_call_id,
            name=tool_name,
            status=tool_status,
            character_count=len(tool_result_text),
        )

    @staticmethod
    def _append_tool_message(
        messages: list,
        content: str,
        tool_call_id: str,
    ) -> None:
        messages.append(
            ToolMessage(
                content=content,
                tool_call_id=tool_call_id,
            )
        )
