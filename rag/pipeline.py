"""把五个步骤串起来。

离线阶段（建索引，文档变了才需要重跑）：
    加载文档 → 切块 → 向量化 → 存入向量库

在线阶段（每次提问）：
    问题 →（多轮时改写问题）→ 问题向量化 → 检索 top_k 块 → 拼 Prompt → Claude 生成回答
"""
import time
from collections.abc import Iterator
from dataclasses import dataclass, field

import anthropic

from . import config
from .chunker import chunk_documents
from .embedder import create_embedder
from .generator import SYSTEM_PROMPT, ClaudeGenerator, build_user_message
from .loader import load_documents
from .vector_store import Hit, VectorStore


def build_index(log=print) -> VectorStore:
    """离线阶段：把 DOCS_DIR 里的文档处理成向量索引并保存。"""
    started = time.time()
    docs = load_documents(config.DOCS_DIR)
    if not docs:
        raise RuntimeError(f"{config.DOCS_DIR} 里没有找到 .txt / .md / .pdf 文档")
    log(f"[1/4] 加载文档：{len({d.source for d in docs})} 个文件，共 {sum(len(d.text) for d in docs)} 字")

    chunks = chunk_documents(docs, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    log(f"[2/4] 切块：{len(chunks)} 块（chunk_size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP}）")

    embedder = create_embedder(config.EMBEDDER, config.EMBED_MODEL)
    texts = [c.text for c in chunks]
    embedder.fit(texts)
    embeddings = embedder.embed_documents(texts)
    log(f"[3/4] 向量化：{config.EMBEDDER}，得到 {embeddings.shape[0]}×{embeddings.shape[1]} 的矩阵")

    meta = {
        "embedder": config.EMBEDDER,
        "embedder_state": embedder.state(),
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "sources": sorted({d.source for d in docs}),
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    store = VectorStore(embeddings, chunks, meta)
    store.save(config.INDEX_DIR)
    log(f"[4/4] 已保存到 {config.INDEX_DIR}（耗时 {time.time() - started:.1f}s）")
    return store


@dataclass
class Conversation:
    """一次多轮对话的状态。

    messages 只往后追加、从不修改已发出的内容：这样 Claude 的提示缓存能命中，
    也符合 API 对历史记录"只追加"的要求。超过 MAX_HISTORY_TURNS 时从最早的一轮整体丢弃。
    """

    messages: list[dict] = field(default_factory=list)
    turns: list[tuple[str, str]] = field(default_factory=list)  # (问题, 回答)，用于改写追问

    def add_turn(self, user_message: str, question: str, answer: str) -> None:
        self.messages += [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": answer},
        ]
        self.turns.append((question, answer))
        if len(self.turns) > config.MAX_HISTORY_TURNS:
            self.turns = self.turns[1:]
            self.messages = self.messages[2:]


class RAGPipeline:
    def __init__(self):
        if not VectorStore.exists(config.INDEX_DIR):
            raise RuntimeError("还没有建索引，请先运行：python ingest.py")
        self.store = VectorStore.load(config.INDEX_DIR)
        kind = self.store.meta["embedder"]
        if kind != config.EMBEDDER:
            print(f"提示：索引是用 {kind} 建的，当前配置是 {config.EMBEDDER}，以索引为准。改了配置请重跑 ingest.py")
        self.embedder = create_embedder(kind, self.store.meta["embedder_state"].get("model", config.EMBED_MODEL))
        self.embedder.load_state(self.store.meta["embedder_state"])
        self._generator: ClaudeGenerator | None = None

    @property
    def generator(self) -> ClaudeGenerator:
        if self._generator is None:  # 用到时才创建，这样只做检索时不需要 API Key
            self._generator = ClaudeGenerator()
        return self._generator

    def reset_generator(self) -> None:
        """API Key 或地址改了之后调用，下次提问时用新设置重新创建客户端。"""
        self._generator = None

    def retrieve(self, query: str, top_k: int = config.TOP_K, min_score: float | None = None) -> list[Hit]:
        if min_score is None:
            min_score = config.MIN_SCORE[self.embedder.name]
        query_vector = self.embedder.embed_query(query)
        return self.store.search(query_vector, top_k, min_score)

    def ask(self, question: str, conv: Conversation) -> Iterator[dict]:
        """在线阶段。以事件流的形式返回每一步的中间结果，方便命令行和网页展示。"""
        try:
            query = question
            if config.QUERY_REWRITE and conv.turns:
                query = self.generator.rewrite_query(question, conv.turns)
                if query != question:
                    yield {"type": "rewrite", "query": query}

            hits = self.retrieve(query)
            yield {
                "type": "sources",
                "hits": [
                    {"index": i, "location": h.chunk.location, "score": round(h.score, 4), "text": h.chunk.text}
                    for i, h in enumerate(hits, start=1)
                ],
            }

            user_message = build_user_message(question, hits)
            yield {"type": "prompt", "system": SYSTEM_PROMPT, "user": user_message}

            answer = ""
            messages = conv.messages + [{"role": "user", "content": user_message}]
            for text in self.generator.stream(messages):
                answer += text
                yield {"type": "text", "text": text}
            conv.add_turn(user_message, question, answer)
            yield {"type": "done"}
        except anthropic.AuthenticationError:
            yield {"type": "error", "message": "API Key 无效：请在网页「设置」或 .env 中检查 ANTHROPIC_API_KEY"}
        except TypeError as e:  # SDK 找不到任何凭据时抛的是 TypeError
            if "authentication" not in str(e):
                raise
            yield {"type": "error", "message": "还没有设置 API Key：请在网页「设置」或 .env 中填写 ANTHROPIC_API_KEY"}
        except anthropic.RateLimitError:
            yield {"type": "error", "message": "请求太频繁（429），稍等几秒再试"}
        except anthropic.APIStatusError as e:
            yield {"type": "error", "message": f"Claude API 返回错误 {e.status_code}：{e.message}"}
        except anthropic.APIConnectionError:
            yield {"type": "error", "message": "连不上 Claude API，请检查网络或代理设置"}
