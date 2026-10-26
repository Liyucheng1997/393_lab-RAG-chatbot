"""第 2 步：切块（Chunking）。

为什么要切块？
  - 一篇长文档里通常只有一小段和问题有关，整篇塞给模型既浪费 token 又会稀释重点；
  - 向量化模型一次能处理的文字有限（bge-small 最多 512 个 token），太长会被截断。

切法：先按句子切开，再把句子一句句拼起来，直到快超过 chunk_size 就收成一块。
不在句子中间硬切，是为了让每块意思尽量完整。

为什么要重叠（overlap）？
  如果答案正好跨在两块的边界上，没有重叠就会被劈成两半，哪块都检索不到完整信息。
  让下一块开头重复上一块末尾的几句话，可以缓解这个问题。
"""
import re
from dataclasses import dataclass

from .loader import Document

# 在中英文句末标点和换行之后断句（标点保留在句子末尾）
_SENTENCE_END = re.compile(r"(?<=[。！？!?；;\n])|(?<=\.)(?=\s)")


@dataclass
class Chunk:
    id: int
    text: str
    source: str
    page: int | None = None

    @property
    def location(self) -> str:
        return f"{self.source} 第{self.page}页" if self.page else self.source


def split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for piece in _SENTENCE_END.split(text):
        if not piece:
            continue
        if piece.strip() or not sentences:
            sentences.append(piece)
        else:
            sentences[-1] += piece  # 单独的换行并到上一句末尾，保留段落格式
    return sentences


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    # 极少数句子本身就超过 chunk_size（比如没有标点的长段落），只能按字数硬切
    pieces: list[str] = []
    for sentence in split_sentences(text):
        if len(sentence) <= chunk_size:
            pieces.append(sentence)
        else:
            step = max(chunk_size - overlap, 1)
            pieces.extend(sentence[i : i + chunk_size] for i in range(0, len(sentence), step))

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for piece in pieces:
        if current and current_len + len(piece) > chunk_size:
            chunks.append("".join(current).strip())
            # 从当前块末尾往回取句子，凑够 overlap 个字作为下一块的开头
            tail: list[str] = []
            tail_len = 0
            for s in reversed(current):
                if tail_len + len(s) > overlap:
                    break
                tail.insert(0, s)
                tail_len += len(s)
            current, current_len = tail, tail_len
        current.append(piece)
        current_len += len(piece)
    if current:
        chunks.append("".join(current).strip())
    return [c for c in chunks if c]


def chunk_documents(docs: list[Document], chunk_size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        for text in split_text(doc.text, chunk_size, overlap):
            chunks.append(Chunk(id=len(chunks), text=text, source=doc.source, page=doc.page))
    return chunks
