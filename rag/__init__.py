"""从零实现的 RAG（检索增强生成）。

    loader.py        第 1 步  加载文档
    chunker.py       第 2 步  切块
    embedder.py      第 3 步  向量化
    vector_store.py  第 4 步  存储与检索
    generator.py     第 5 步  拼 Prompt + 调用 Claude 生成
    pipeline.py      把以上步骤串起来
"""

__version__ = "1.0.0"
