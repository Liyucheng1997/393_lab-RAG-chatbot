"""建索引：读取 data/docs 里的文档，切块、向量化，保存到 data/index。

用法：
    python ingest.py               # 用 .env 里配置的向量化方式（默认 bge）
    python ingest.py --tfidf       # 临时改用手写 TF-IDF，不需要下载模型
"""
import argparse

from rag import config
from rag.pipeline import build_index

parser = argparse.ArgumentParser(description="构建 RAG 向量索引")
parser.add_argument("--tfidf", action="store_true", help="使用手写 TF-IDF 关键词向量")
args = parser.parse_args()
if args.tfidf:
    config.EMBEDDER = "tfidf"

build_index()
