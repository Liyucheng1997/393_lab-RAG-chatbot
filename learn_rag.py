"""RAG 原理逐步演示：把每一步的中间结果打印出来。

用法：
    python learn_rag.py                         # 用默认问题演示
    python learn_rag.py "员工年假有几天？"       # 换成你自己的问题
    python learn_rag.py --no-llm                # 只演示检索，不调用 Claude（不需要 API Key）
"""
import argparse
import sys

import numpy as np

from rag import config
from rag.chunker import chunk_documents
from rag.embedder import TfidfEmbedder, create_embedder, tokenize
from rag.generator import SYSTEM_PROMPT, ClaudeGenerator, build_user_message
from rag.loader import load_documents
from rag.vector_store import VectorStore

parser = argparse.ArgumentParser()
# 默认问题故意用"积分"，而文档里的说法是"星光值"——看看两种检索谁能找到
parser.add_argument("question", nargs="?", default="会员积分能换什么东西？")
parser.add_argument("--no-llm", action="store_true", help="跳过第 5 步，不调用 Claude")
args = parser.parse_args()
sys.stdout.reconfigure(encoding="utf-8")


def title(text: str) -> None:
    print(f"\n\033[94m{'=' * 60}\n{text}\n{'=' * 60}\033[0m")


def show_hits(hits) -> None:
    for rank, hit in enumerate(hits, start=1):
        preview = hit.chunk.text.replace("\n", " ")[:50]
        print(f"  第{rank}名  相似度 {hit.score:.3f}  [{hit.chunk.location}] {preview}…")


question = args.question
print(f"演示问题：{question}")

# ---------------------------------------------------------------- 第 1 步
title("第 1 步  加载文档：把文件读成纯文本")
docs = load_documents(config.DOCS_DIR)
for d in docs:
    print(f"  {d.source}{f' 第{d.page}页' if d.page else ''}：{len(d.text)} 字")

# ---------------------------------------------------------------- 第 2 步
title(f"第 2 步  切块：每块最多 {config.CHUNK_SIZE} 字，相邻块重叠约 {config.CHUNK_OVERLAP} 字")
chunks = chunk_documents(docs, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
print(f"  共切成 {len(chunks)} 块。看看前两块（注意第 2 块开头和第 1 块结尾的重叠）：\n")
for c in chunks[:2]:
    print(f"  ── 块 {c.id}（{len(c.text)} 字）──\n  {c.text.replace(chr(10), chr(10) + '  ')}\n")

# ---------------------------------------------------------------- 第 3 步
title("第 3 步  向量化：把文字变成数字向量")
texts = [c.text for c in chunks]

tfidf = TfidfEmbedder()
tfidf.fit(texts)
tfidf_vectors = tfidf.embed_documents(texts)
print(f"  [TF-IDF] 问题被切成这些词：{tokenize(question)[:20]} …")
print(f"  [TF-IDF] 词表大小 = 向量维度 = {tfidf_vectors.shape[1]}，绝大多数维度是 0（稀疏向量）")

bge = None
try:
    bge = create_embedder("bge", config.EMBED_MODEL)
    bge_vectors = bge.embed_documents(texts)
    q = bge.embed_query(question)
    print(f"  [BGE]    向量维度 = {bge_vectors.shape[1]}（稠密向量），问题向量的前 8 维：")
    print(f"           [{', '.join(f'{x:.4f}' for x in q[:8])}, …]")
    print(f"           向量长度（L2 范数）= {np.linalg.norm(q):.4f}，已归一化，所以点积 = 余弦相似度")
except ImportError:
    print("  [BGE]    未安装 sentence-transformers，跳过语义向量演示（pip install sentence-transformers）")

# ---------------------------------------------------------------- 第 4 步
title(f"第 4 步  检索：算问题和每一块的相似度，取前 {config.TOP_K} 名")
print("  [TF-IDF 关键词检索] 只看字面重合：")
tfidf_hits = VectorStore(tfidf_vectors, chunks, {}).search(tfidf.embed_query(question), config.TOP_K)
show_hits(tfidf_hits)
hits, kind = tfidf_hits, "tfidf"
if bge is not None:
    kind = "bge"
    print("\n  [BGE 语义检索] 能理解同义表达：")
    hits = VectorStore(bge_vectors, chunks, {}).search(bge.embed_query(question), config.TOP_K)
    show_hits(hits)
    print("\n  对比一下两种检索的排名：问题里的说法和文档原文不同时，语义检索通常更准。")

# ---------------------------------------------------------------- 第 5 步
hits = [h for h in hits if h.score >= config.MIN_SCORE[kind]]
title("第 5 步  增强生成：把检索结果拼进 Prompt，交给 Claude 回答")
print(f"  （相似度低于 {config.MIN_SCORE[kind]} 的块被丢弃，剩 {len(hits)} 块）\n")
user_message = build_user_message(question, hits)
print(f"  ── System Prompt ──\n{SYSTEM_PROMPT}\n")
print(f"  ── User 消息 ──\n{user_message}\n")

if args.no_llm:
    print("（已跳过调用 Claude）")
else:
    print("  ── Claude 的回答 ──")
    for text in ClaudeGenerator().stream([{"role": "user", "content": user_message}]):
        print(text, end="", flush=True)
    print()
