"""第 5 步：增强生成（Augmented Generation）。

把检索到的文档块拼进 Prompt，让 Claude "开卷答题"：
  - System Prompt 规定规则：只依据给定资料回答、标注引用编号、资料里没有就说不知道；
  - User 消息 = <documents>资料</documents> + 用户问题。

这一步决定了 RAG 能不能"不胡说"：模型本身的知识可能过时或根本不知道你的私有文档，
但只要正确的资料被放进了上下文，它就能据此作答，并给出可核对的出处。
"""
from collections.abc import Iterator

import anthropic

from . import config
from .vector_store import Hit

SYSTEM_PROMPT = """你是一个知识库问答助手。用户每次提问时，系统会附上从知识库中检索到的资料片段，放在 <documents> 标签里，每个片段有编号。

回答规则：
1. 只依据 <documents> 中的资料回答。资料里没有的信息，不要用自己的知识补充或猜测。
2. 资料不足以回答时，直接说明"知识库中没有找到相关信息"，可以说明资料里有哪些相近内容。
3. 引用资料时在句末标注片段编号，例如 [1] 或 [2][3]。
4. 用用户提问的语言回答，简洁清楚，条理分明。"""

REWRITE_PROMPT = """下面是一段对话记录和用户的最新问题。最新问题可能省略了主语或用代词指代前文（比如"它""这个""那第二点呢"）。
请把最新问题改写成一个不依赖上下文、可以单独拿去检索的完整问题。只输出改写后的问题本身，不要解释。

<history>
{history}
</history>

最新问题：{question}"""


def format_context(hits: list[Hit]) -> str:
    if not hits:
        return "<documents>\n（没有检索到相关资料）\n</documents>"
    parts = [
        f'<document index="{i}" source="{hit.chunk.location}">\n{hit.chunk.text}\n</document>'
        for i, hit in enumerate(hits, start=1)
    ]
    return "<documents>\n" + "\n".join(parts) + "\n</documents>"


def build_user_message(question: str, hits: list[Hit]) -> str:
    return f"{format_context(hits)}\n\n问题：{question}"


class ClaudeGenerator:
    def __init__(self):
        # Key 来自网页「设置」或 .env；传 None 时 SDK 会自己去找环境变量
        self.client = anthropic.Anthropic(api_key=config.API_KEY or None, base_url=config.BASE_URL or None)

    def _request_args(self, messages: list[dict], system: str | None) -> dict:
        args = dict(
            model=config.CLAUDE_MODEL,
            max_tokens=config.MAX_TOKENS,
            messages=messages,
            output_config={"effort": config.EFFORT},
        )
        if system:
            args["system"] = system
        if config.USE_FALLBACKS:
            # 主模型因安全策略拒答时，由服务端自动换模型重答（同一次请求内完成）
            args["betas"] = ["server-side-fallback-2026-07-01"]
            args["fallbacks"] = "default"
        return args

    def stream(self, messages: list[dict]) -> Iterator[str]:
        """流式生成：一边生成一边吐出文字片段。"""
        args = self._request_args(messages, SYSTEM_PROMPT)
        with self.client.beta.messages.stream(**args) as stream:
            yield from stream.text_stream
            final = stream.get_final_message()
        if final.stop_reason == "refusal":
            yield "\n\n（模型拒绝回答了这个问题。）"
        elif final.stop_reason == "max_tokens":
            yield "\n\n（回答达到长度上限，被截断了。）"

    def rewrite_query(self, question: str, history: list[tuple[str, str]]) -> str:
        """多轮对话时，把"它的缺点呢？"这类追问改写成完整问题，否则检索会找不到东西。"""
        history_text = "\n".join(f"用户：{q}\n助手：{a}" for q, a in history)
        prompt = REWRITE_PROMPT.format(history=history_text, question=question)
        args = self._request_args([{"role": "user", "content": prompt}], system=None)
        args["max_tokens"] = 1024
        args["output_config"] = {"effort": "low"}
        response = self.client.beta.messages.create(**args)
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        return text or question
