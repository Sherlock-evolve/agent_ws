import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


load_dotenv()

model = ChatOpenAI(
    model=os.getenv("ZHIPU_MODEL"),
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url=os.getenv("ZHIPU_BASE_URL"),
    temperature=0.7,
)

messages = [
    SystemMessage(
        content="你是一位人工智能老师，请用通俗、准确的方式回答。"
    )
]

while True:
    question = input("\n你：").strip()

    if question.lower() in {"exit", "quit", "退出"}:
        break

    if not question:
        continue

    messages.append(HumanMessage(content=question))

    print("\nAI：", end="", flush=True)
    response_parts = []
    for chunk in model.stream(messages):
        if chunk.content:
            print(chunk.content, end="", flush=True)
            response_parts.append(chunk.content)
    print()

    messages.append(AIMessage(content="".join(response_parts)))
