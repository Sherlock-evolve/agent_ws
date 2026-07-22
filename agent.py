import json
from dataclasses import dataclass, field

from langchain_core.messages import (
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


@dataclass
class _TurnState:
    tool_call_count: int = 0
    seen_tool_calls: set[tuple[str, str]] = field(default_factory=set)
    tool_budget_exhausted: bool = False


@dataclass(frozen=True)
class AgentEvent:
    kind: str
    text: str | None = None
    step: int | None = None
    tool_name: str | None = None
    args_text: str | None = None
    succeeded: bool | None = None
    result_length: int | None = None


class WorkspaceAgent:
    def __init__(
        self,
        model,
        tools,
        max_agent_loops=5,
        max_tool_calls=8,
        event_handler=None,
    ):
        self.model = model
        self.tools = list(tools)
        self.max_agent_loops = max_agent_loops
        self.max_tool_calls = max_tool_calls
        self.tools_by_name = {tool.name: tool for tool in self.tools}
        self.model_with_tools = self.model.bind_tools(self.tools)
        self.messages = [SystemMessage(content=SYSTEM_PROMPT)]
        self.event_handler = event_handler

    def run_turn(self, question: str) -> None:
        self.messages.append(HumanMessage(content=question))

        state = _TurnState()
        answered = False
        task_stopped = False

        for step in range(1, self.max_agent_loops + 1):
            tools_allowed = (
                step < self.max_agent_loops
                and not state.tool_budget_exhausted
            )
            active_model = self.model_with_tools if tools_allowed else self.model
            response = self._stream_response(active_model)

            if response is None:
                self._emit(
                    AgentEvent(
                        kind="system",
                        text="模型未返回任何消息，当前任务已停止。",
                    )
                )
                task_stopped = True
                break

            self.messages.append(response)

            if not response.tool_calls:
                answered = True
                break

            for tool_call in response.tool_calls:
                self._execute_tool_call(
                    tool_call=tool_call,
                    step=step,
                    tools_allowed=tools_allowed,
                    state=state,
                )

        if not answered and not task_stopped:
            self._emit(
                AgentEvent(
                    kind="system",
                    text=(
                        f"Agent 循环达到 {self.max_agent_loops} 次上限，"
                        "已停止。"
                    ),
                )
            )

    def _stream_response(self, active_model):
        response_chunk = None

        for chunk in active_model.stream(self.messages):
            if chunk.content:
                self._emit(AgentEvent(kind="text_delta", text=chunk.content))

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
    ) -> None:
        tool_name = tool_call["name"]
        signature = (
            tool_name,
            json.dumps(
                tool_call["args"],
                ensure_ascii=False,
                sort_keys=True,
            ),
        )
        args_text = signature[1]
        self._emit(
            AgentEvent(
                kind="tool_call",
                step=step,
                tool_name=tool_name,
                args_text=args_text,
            )
        )

        if not tools_allowed:
            self._emit(
                AgentEvent(
                    kind="tool_skip",
                    text="当前轮次禁止调用工具",
                )
            )
            self._append_tool_message(
                content=(
                    "当前轮次不允许调用工具，本次调用未执行。"
                    "请根据已有信息直接回答。"
                ),
                tool_call_id=tool_call["id"],
            )
            return

        if signature in state.seen_tool_calls:
            self._emit(AgentEvent(kind="tool_skip", text="重复调用"))
            self._append_tool_message(
                content="重复工具调用已跳过，请使用之前相同工具调用的结果。",
                tool_call_id=tool_call["id"],
            )
            return

        state.seen_tool_calls.add(signature)

        if state.tool_call_count >= self.max_tool_calls:
            state.tool_budget_exhausted = True
            self._emit(AgentEvent(kind="tool_skip", text="工具预算已耗尽"))
            self._append_tool_message(
                content=(
                    "工具预算已耗尽，本次调用未执行。"
                    "请根据已有信息回答。"
                ),
                tool_call_id=tool_call["id"],
            )
            return

        state.tool_call_count += 1
        if state.tool_call_count >= self.max_tool_calls:
            state.tool_budget_exhausted = True

        try:
            selected_tool = self.tools_by_name[tool_name]
            tool_result_text = str(selected_tool.invoke(tool_call["args"]))
            tool_succeeded = True
        except Exception as error:
            tool_result_text = f"工具执行失败：{error}"
            tool_succeeded = False

        result_status = "成功" if tool_succeeded else "失败"
        self._emit(
            AgentEvent(
                kind="tool_result",
                text=result_status,
                succeeded=tool_succeeded,
                result_length=len(tool_result_text),
            )
        )
        self._append_tool_message(
            content=tool_result_text,
            tool_call_id=tool_call["id"],
        )

    def _append_tool_message(self, content: str, tool_call_id: str) -> None:
        self.messages.append(
            ToolMessage(
                content=content,
                tool_call_id=tool_call_id,
            )
        )

    def _emit(self, event: AgentEvent) -> None:
        if self.event_handler is not None:
            self.event_handler(event)
