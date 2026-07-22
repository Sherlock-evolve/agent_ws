import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from agent import AgentEvent, WorkspaceAgent
from tools import list_files, read_file, search_text


class CliRenderer:
    def __init__(self):
        self.ai_label_printed = False
        self.cursor_at_line_start = True

    def start_turn(self) -> None:
        print()
        self.ai_label_printed = False
        self.cursor_at_line_start = True

    def finish_turn(self) -> None:
        self._ensure_line_start()

    def handle(self, event: AgentEvent) -> None:
        if event.kind == "text_delta":
            if not self.ai_label_printed:
                print("AI：", end="", flush=True)
                self.ai_label_printed = True
            print(event.text, end="", flush=True)
            self.cursor_at_line_start = event.text.endswith("\n")
            return

        self._ensure_line_start()

        if event.kind == "tool_call":
            print(
                f"[第 {event.step} 轮工具调用] "
                f"{event.tool_name} {event.args_text}"
            )
        elif event.kind == "tool_skip":
            print(f"[工具跳过] {event.text}")
        elif event.kind == "tool_result":
            print(
                f"[工具结果] {event.text}，"
                f"返回 {event.result_length} 个字符"
            )
        elif event.kind == "system":
            print(f"[系统] {event.text}")

    def _ensure_line_start(self) -> None:
        if not self.cursor_at_line_start:
            print()
            self.cursor_at_line_start = True


def main() -> None:
    load_dotenv()

    model = ChatOpenAI(
        model=os.getenv("ZHIPU_MODEL"),
        api_key=os.getenv("ZHIPU_API_KEY"),
        base_url=os.getenv("ZHIPU_BASE_URL"),
        temperature=0,
    )
    renderer = CliRenderer()
    agent = WorkspaceAgent(
        model=model,
        tools=[list_files, read_file, search_text],
        event_handler=renderer.handle,
    )

    while True:
        question = input("\n你：").strip()

        if question.lower() in {"exit", "quit", "退出"}:
            break

        if not question:
            continue

        renderer.start_turn()
        agent.run_turn(question)
        renderer.finish_turn()


if __name__ == "__main__":
    main()
