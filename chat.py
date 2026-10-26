"""命令行聊天。

用法：
    python chat.py           # 普通聊天，回答后列出引用来源
    python chat.py --debug   # 学习模式：额外打印检索分数和完整 Prompt
"""
import argparse
import sys

from rag.pipeline import Conversation, RAGPipeline

parser = argparse.ArgumentParser(description="RAG 命令行聊天")
parser.add_argument("--debug", action="store_true", help="打印检索结果和发给 Claude 的完整 Prompt")
args = parser.parse_args()

sys.stdout.reconfigure(encoding="utf-8")
GRAY, BLUE, RED, RESET = "\033[90m", "\033[94m", "\033[91m", "\033[0m"

pipeline = RAGPipeline()
conv = Conversation()
print(f"知识库已加载：{len(pipeline.store.chunks)} 块，来自 {len(pipeline.store.meta['sources'])} 个文件。")
print("输入问题开始对话；输入 /new 开始新对话，/quit 退出。\n")

while True:
    try:
        question = input(f"{BLUE}你：{RESET}").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not question:
        continue
    if question == "/quit":
        break
    if question == "/new":
        conv = Conversation()
        print("已开始新对话。\n")
        continue

    sources = []
    for event in pipeline.ask(question, conv):
        kind = event["type"]
        if kind == "rewrite":
            print(f"{GRAY}（追问改写为：{event['query']}）{RESET}")
        elif kind == "sources":
            sources = event["hits"]
            if args.debug:
                print(f"{GRAY}── 检索结果 ──")
                for h in sources:
                    preview = h["text"].replace("\n", " ")[:60]
                    print(f"[{h['index']}] 相似度 {h['score']:.3f}  {h['location']}  {preview}…")
                print(RESET, end="")
        elif kind == "prompt":
            if args.debug:
                print(f"{GRAY}── 发给 Claude 的 User 消息 ──\n{event['user']}\n──────────{RESET}")
            print(f"{BLUE}Claude：{RESET}", end="", flush=True)
        elif kind == "text":
            print(event["text"], end="", flush=True)
        elif kind == "error":
            print(f"{RED}{event['message']}{RESET}")
        elif kind == "done":
            print()
            if sources:
                print(GRAY + "来源：" + "；".join(f"[{h['index']}] {h['location']}" for h in sources) + RESET)
    print()
