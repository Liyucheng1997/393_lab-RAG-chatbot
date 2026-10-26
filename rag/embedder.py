"""第 3 步：向量化（Embedding）。

把一段文字变成一个固定长度的数字向量，让"意思相近的文字"在向量空间里"距离也相近"。
之后检索就变成了纯数学：算问题向量和每个块向量的相似度，取最高的几个。

这里实现了两种向量化方式，方便对比：

1. TfidfEmbedder —— 关键词向量（纯 numpy 手写，不依赖任何模型）
   每一维对应词表里的一个"词"，值 = 这个词在本段出现的次数 × 这个词的稀有程度(IDF)。
   优点：完全透明、零下载。缺点：只认字面，"年假"和"带薪休假"它会认为毫不相关。

2. BgeEmbedder —— 语义向量（神经网络模型 BAAI/bge-small-zh-v1.5）
   模型在海量句子对上训练过，能把同义表达映射到相近的位置，512 维。
   这是实际 RAG 系统里的标准做法。

两种向量都做了 L2 归一化（长度为 1），所以"点积 = 余弦相似度"，范围 [-1, 1]。
"""
import math
import re
from collections import Counter

import numpy as np


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


# ---------------------------------------------------------------------------
# 方式 1：TF-IDF（手写）
# ---------------------------------------------------------------------------

_TOKEN = re.compile(r"[a-z0-9]+|[一-鿿]")


def tokenize(text: str) -> list[str]:
    """中文没有空格分词，这里用"单字 + 相邻两字"来近似词语；英文按单词切。"""
    tokens = _TOKEN.findall(text.lower())
    bigrams = [
        a + b
        for a, b in zip(tokens, tokens[1:])
        if len(a) == 1 and len(b) == 1 and "一" <= a <= "鿿" and "一" <= b <= "鿿"
    ]
    return tokens + bigrams


class TfidfEmbedder:
    name = "tfidf"

    def __init__(self):
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray = np.zeros(0)

    def fit(self, texts: list[str]) -> None:
        """统计词表和每个词的 IDF。IDF 越大说明这个词越稀有、越有区分度。"""
        doc_freq: Counter = Counter()
        for text in texts:
            doc_freq.update(set(tokenize(text)))
        self.vocab = {tok: i for i, tok in enumerate(sorted(doc_freq))}
        n = len(texts)
        self.idf = np.array(
            [math.log((1 + n) / (1 + doc_freq[tok])) + 1 for tok in sorted(doc_freq)],
            dtype=np.float32,
        )

    def _transform(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), len(self.vocab)), dtype=np.float32)
        for row, text in enumerate(texts):
            for tok, count in Counter(tokenize(text)).items():
                col = self.vocab.get(tok)
                if col is not None:  # 词表里没有的词（训练时没见过）直接忽略
                    vectors[row, col] = count * self.idf[col]
        return l2_normalize(vectors)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._transform(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._transform([text])[0]

    # 词表和 IDF 是从知识库里统计出来的，必须跟索引一起保存
    def state(self) -> dict:
        return {"vocab": self.vocab, "idf": self.idf.tolist()}

    def load_state(self, state: dict) -> None:
        self.vocab = state["vocab"]
        self.idf = np.array(state["idf"], dtype=np.float32)


# ---------------------------------------------------------------------------
# 方式 2：BGE 语义向量
# ---------------------------------------------------------------------------


class BgeEmbedder:
    name = "bge"

    # BGE 的约定：问题前面加这句指令，文档不加。
    # 因为"问题"和"答案段落"长得很不一样（不对称检索），加指令能让问题向量更靠近答案。
    QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def fit(self, texts: list[str]) -> None:
        pass  # 预训练模型，不需要在你的数据上统计任何东西

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=len(texts) > 64)

    def embed_query(self, text: str) -> np.ndarray:
        return self.model.encode(self.QUERY_INSTRUCTION + text, normalize_embeddings=True)

    def state(self) -> dict:
        return {"model": self.model_name}

    def load_state(self, state: dict) -> None:
        pass


def create_embedder(kind: str, model_name: str):
    if kind == "tfidf":
        return TfidfEmbedder()
    if kind == "bge":
        return BgeEmbedder(model_name)
    raise ValueError(f"未知的 EMBEDDER：{kind}（可选 bge / tfidf）")
