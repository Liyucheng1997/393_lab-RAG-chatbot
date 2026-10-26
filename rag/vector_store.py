"""第 4 步：向量库与检索。

所谓"向量数据库"，最朴素的形态就是一个 N×D 的矩阵（N 个块，每块 D 维）。
检索 = 问题向量和矩阵每一行做点积，排序取前 top_k。

Chroma、Milvus、FAISS 等专业向量库做的事情本质一样，只是在百万级数据上
用近似最近邻（ANN）索引把"逐个比较"加速了。几千块的知识库，numpy 一行矩阵乘法就够了。
"""
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .chunker import Chunk


@dataclass
class Hit:
    chunk: Chunk
    score: float  # 余弦相似度，越接近 1 越相关


class VectorStore:
    def __init__(self, embeddings: np.ndarray, chunks: list[Chunk], meta: dict):
        self.embeddings = embeddings.astype(np.float32)
        self.chunks = chunks
        self.meta = meta  # 记录建索引时用的向量化方式、切块参数等

    def search(self, query_vector: np.ndarray, top_k: int, min_score: float = -1.0) -> list[Hit]:
        # 向量都已归一化，所以点积就是余弦相似度
        scores = self.embeddings @ query_vector
        best = np.argsort(-scores)[:top_k]
        return [Hit(self.chunks[i], float(scores[i])) for i in best if scores[i] >= min_score]

    def save(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        np.save(index_dir / "embeddings.npy", self.embeddings)
        (index_dir / "chunks.json").write_text(
            json.dumps([asdict(c) for c in self.chunks], ensure_ascii=False, indent=1), encoding="utf-8"
        )
        (index_dir / "meta.json").write_text(json.dumps(self.meta, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, index_dir: Path) -> "VectorStore":
        embeddings = np.load(index_dir / "embeddings.npy")
        chunks = [Chunk(**c) for c in json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))]
        meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
        return cls(embeddings, chunks, meta)

    @staticmethod
    def exists(index_dir: Path) -> bool:
        return (index_dir / "meta.json").exists()
