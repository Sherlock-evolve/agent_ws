import argparse
import json
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from agent import WorkspaceAgent
from contracts import (
    AgentEvent,
    ApprovalDecision,
    ApprovalRequiredEvent,
    ContextTrimmedEvent,
    MemoryUpdatedEvent,
    SessionSavedEvent,
    SystemEvent,
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from persistent_session import (
    PersistentSession,
    PersistentSessionSaveError,
)
from tools import (
    list_files,
    prepare_write_file,
    read_file,
    search_text,
    write_file,
)


ai_label_printed = False
cursor_at_line_start = True
STATUS_LABELS = {
    "success": "成功",
    "error": "失败",
}
EXIT_COMMANDS = {"exit", "quit", "退出"}


class _ExitRequested(Exception):
    pass


def start_turn() -> None:
    global ai_label_printed, cursor_at_line_start

    print()
    ai_label_printed = False
    cursor_at_line_start = True


def finish_turn() -> None:
    _ensure_line_start()


def render_event(event: AgentEvent) -> None:
    global ai_label_printed, cursor_at_line_start

    if isinstance(event, TokenEvent):
        if not ai_label_printed:
            print("AI：", end="", flush=True)
            ai_label_printed = True
        print(event.text, end="", flush=True)
        cursor_at_line_start = event.text.endswith("\n")
        return

    _ensure_line_start()

    if isinstance(event, ToolCallEvent):
        args_text = json.dumps(
            event.args,
            ensure_ascii=False,
            sort_keys=True,
        )
        print(f"[第 {event.step} 轮工具调用] {event.name} {args_text}")
    elif isinstance(event, ToolResultEvent):
        if event.status == "skipped":
            print(f"[工具跳过] {event.detail}")
        else:
            status_label = STATUS_LABELS[event.status]
            truncated_label = "（已截断）" if event.truncated else ""
            print(
                f"[工具结果] {status_label}，"
                f"返回 {event.character_count} 个字符"
                f"{truncated_label}"
            )
    elif isinstance(event, ApprovalRequiredEvent):
        print(f"[审批] 工具 {event.tool_name} 等待用户确认")
        if event.preview:
            print(event.preview)
    elif isinstance(event, SystemEvent):
        print(f"[系统] {event.message}")
    elif isinstance(event, ContextTrimmedEvent):
        print(f"[上下文] 已移除 {event.removed_message_count} 条旧消息")
    elif isinstance(event, MemoryUpdatedEvent):
        print(f"[记忆] 长期摘要已更新，共 {event.character_count} 个字符")
    elif isinstance(event, SessionSavedEvent):
        print(f"[会话] 已保存：{event.session_id}")


def _ensure_line_start() -> None:
    global cursor_at_line_start

    if not cursor_at_line_start:
        print()
        cursor_at_line_start = True


def create_workspace_agent() -> WorkspaceAgent:
    model = ChatOpenAI(
        model=os.getenv("ZHIPU_MODEL"),
        api_key=os.getenv("ZHIPU_API_KEY"),
        base_url=os.getenv("ZHIPU_BASE_URL"),
        temperature=0,
    )
    agent = WorkspaceAgent(
        model=model,
        tools=[list_files, read_file, search_text, write_file],
        approval_required_tools={write_file.name},
        approval_preparers={write_file.name: prepare_write_file},
    )
    return agent


def _drive_turn(
    session: PersistentSession,
    question: str,
) -> None:
    start_turn()
    stream = session.stream_turn(question)
    decision = None

    try:
        while True:
            try:
                event = stream.send(decision)
            except StopIteration:
                break

            decision = None
            render_event(event)

            if isinstance(event, ApprovalRequiredEvent):
                answer = input("是否允许执行？[y/N] ")
                normalized_answer = answer.strip().lower()
                if normalized_answer in EXIT_COMMANDS:
                    raise _ExitRequested
                decision = ApprovalDecision(
                    tool_call_id=event.tool_call_id,
                    approved=normalized_answer in {"y", "yes"},
                )
    except (EOFError, KeyboardInterrupt, _ExitRequested):
        raise
    except PersistentSessionSaveError as error:
        _ensure_line_start()
        print(f"[保存失败] {error}")
    except Exception as error:
        _ensure_line_start()
        print(f"[运行失败] {error}")
    finally:
        stream.close()
        finish_turn()


def _exit_status(
    session: PersistentSession,
    *,
    interrupted: bool = False,
) -> int:
    if session.dirty:
        print("[警告] 当前会话仍有未保存状态")
        if not interrupted:
            return 1
    return 130 if interrupted else 0


def run_cli(session: PersistentSession) -> int:
    try:
        while True:
            question = input("\n你：").strip()

            if question.lower() in EXIT_COMMANDS:
                return _exit_status(session)

            if not question:
                continue

            if question == ":retry":
                if not session.dirty:
                    print("[会话] 当前没有未保存状态")
                    continue
                try:
                    session.flush()
                except PersistentSessionSaveError as error:
                    print(f"[保存失败] {error}")
                except Exception as error:
                    print(f"[保存失败] {error}")
                else:
                    print(f"[会话] 重试保存成功：{session.session_id}")
                continue

            if session.dirty:
                print("[会话] 存在未保存状态，请先输入 :retry 或退出")
                continue

            if question.startswith(":"):
                print(f"[命令] 未知命令：{question}")
                continue

            _drive_turn(session, question)
    except EOFError:
        return _exit_status(session)
    except KeyboardInterrupt:
        print()
        return _exit_status(session, interrupted=True)
    except _ExitRequested:
        return _exit_status(session)


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="工作区 Agent",
    )
    parser.add_argument(
        "--session",
        default="default",
        help="持久会话 ID（默认：default）",
    )
    return parser


def main(argv=None) -> int:
    arguments = _build_argument_parser().parse_args(argv)
    load_dotenv()

    try:
        session = PersistentSession.open(
            arguments.session,
            create_workspace_agent,
        )
    except Exception as error:
        print(f"[启动失败] {error}")
        return 2

    return run_cli(session)


if __name__ == "__main__":
    raise SystemExit(main())
