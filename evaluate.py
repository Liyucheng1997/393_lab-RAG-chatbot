"""检索效果评测：不调用大模型，只看"该找到的片段有没有被找到"。

RAG 答错，十有八九是检索这一步就没找到正确资料。所以调参（切块大小、向量模型、top_k）
之前，先要有一个能量化检索好坏的办法：

    Hit@K：正确片段出现在前 K 个结果里的问题占比（K = TOP_K，因为只有前 K 个会给模型看）
    MRR  ：正确片段排名的倒数取平均（排第 1 得 1 分，第 2 得 0.5，第 3 得 0.33…），越高越好

测试集在 data/eval_questions.json：每题写上问题和"正确片段里一定包含的一句原文"。

用法：
    python evaluate.py                        # 评测当前配置
    python evaluate.py --compare              # 对比多组切块参数 × 两种向量化方式
"""
import argparse
import json
import sys

from rag import config
from rag.chunker import chunk_documents
from rag.embedder import create_embedder
from rag.loader import load_documents
from rag.vector_store import VectorStore

parser = argparse.ArgumentParser()
parser.add_argument("--compare", action="store_true", help="对比多组参数")
args = parser.parse_args()
sys.stdout.reconfigure(encoding="utf-8")

cases = json.loads((config.ROOT / "data" / "eval_questions.json").read_text(encoding="utf-8"))
docs = load_documents(config.DOCS_DIR)
embedders = {}


def evaluate(kind: str, chunk_size: int, overlap: int, verbose: bool = False) -> tuple[float, float, int]:
    if kind not in embedders:
        embedders[kind] = create_embedder(kind, config.EMBED_MODEL)
    embedder = embedders[kind]
    chunks = chunk_documents(docs, chunk_size, overlap)
    texts = [c.text for c in chunks]
    embedder.fit(texts)
    store = VectorStore(embedder.embed_documents(texts), chunks, {})

    hits_at_k, reciprocal_ranks = 0, 0.0
    for case in cases:
        results = store.search(embedder.embed_query(case["question"]), top_k=len(chunks))
        rank = next((i for i, h in enumerate(results, start=1) if case["answer_contains"] in h.chunk.text), None)
        if rank and rank <= config.TOP_K:
            hits_at_k += 1
        reciprocal_ranks += 1 / rank if rank else 0
        if verbose:
            mark = "✓" if rank and rank <= config.TOP_K else "✗"
            print(f"  {mark} 第 {rank or '-':>2} 名  {case['question']}")
    return hits_at_k / len(cases), reciprocal_ranks / len(cases), len(chunks)


if args.compare:
    print(f"共 {len(cases)} 道题，Hit@{config.TOP_K} 和 MRR 都是越高越好\n")
    print(f"{'向量化':<8}{'chunk_size':>11}{'overlap':>9}{'块数':>6}{f'Hit@{config.TOP_K}':>9}{'MRR':>8}")
    for kind in ("tfidf", "bge"):
        for size, overlap in [(100, 20), (150, 30), (200, 40), (300, 60), (500, 100)]:
            hit, mrr, n = evaluate(kind, size, overlap)
            print(f"{kind:<8}{size:>11}{overlap:>9}{n:>6}{hit:>9.0%}{mrr:>8.3f}")
else:
    print(f"当前配置：{config.EMBEDDER}，chunk_size={config.CHUNK_SIZE}，overlap={config.CHUNK_OVERLAP}，top_k={config.TOP_K}\n")
    hit, mrr, n = evaluate(config.EMBEDDER, config.CHUNK_SIZE, config.CHUNK_OVERLAP, verbose=True)
    print(f"\n共 {n} 块。Hit@{config.TOP_K} = {hit:.0%}，MRR = {mrr:.3f}")
