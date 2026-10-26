# RAG 聊天机器人（从零手写版）

一个用来**学习 RAG 原理**的知识库问答机器人。核心流程全部手写（切块、向量化、余弦相似度检索、拼 Prompt），不依赖 LangChain、LlamaIndex 这类框架，每一步的中间结果都能打印出来看。生成回答用的是 Claude（`claude-opus-5-5`）。

原理讲解见 [docs/RAG原理笔记.md](docs/RAG原理笔记.md)。

## 快速开始

```bash
# 1. 安装依赖（Python 3.10+）
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt     # macOS / Linux：.venv/bin/python

# 2. 建索引（把 data/docs 里的文档切块、向量化）
.venv\Scripts\python ingest.py

# 3. 启动网页版，打开 http://127.0.0.1:8000
.venv\Scripts\python app.py
```

第一次打开网页会弹出「设置」，填入 Anthropic API Key 即可开始提问。在设置里还可以：

- **测试连接**：查询一次模型信息来验证 Key、地址和模型名，不消耗 token；
- **接口地址**：使用代理或中转服务时填写，留空就是官方地址；
- **模型 / 思考深度**：默认 `claude-opus-5-5`、思考深度「低」，也可以选别的模型或自定义模型名。

设置保存在项目根目录的 `.env` 文件里，只存在你自己的电脑上，完整的 Key 不会回传给浏览器。`.env` 已加入 `.gitignore`，不会被提交。

除了网页版，还有两个命令行工具（读取同一个 `.env`，也可以照着 `.env.example` 手动写）：

```bash
.venv\Scripts\python chat.py --debug   # 命令行聊天，--debug 会打印检索结果和完整 Prompt
.venv\Scripts\python learn_rag.py      # 逐步演示 RAG 的五个步骤（加 --no-llm 不需要 Key）
```

首次运行会自动下载向量模型 `BAAI/bge-small-zh-v1.5`（约 95MB）。下载慢的话在 `.env` 里加上 `HF_ENDPOINT=https://hf-mirror.com`；不想下载可以用 `python ingest.py --tfidf` 改用手写的 TF-IDF 向量。

## 推荐的学习顺序

1. `python learn_rag.py --no-llm`：不需要 API Key，看清"加载 → 切块 → 向量化 → 检索 → 拼 Prompt"每一步输出了什么。注意默认问题用的是"积分"，文档里写的是"星光值"，对比一下两种检索的结果。
2. 按编号读 `rag/` 下的代码，每个文件对应一步，注释里写了为什么这么做。
3. 填好 API Key，运行 `python chat.py --debug`，问几个问题，再问一个知识库里没有的（比如"公司股价多少"），看模型会不会编造。
4. `python evaluate.py --compare`：改切块大小、换向量化方式，用 Hit@K 和 MRR 量化检索效果。
5. 把你自己的文档放进 `data/docs/`（或者在网页里上传），重建索引后提问。

## 项目结构

```
rag/
  config.py        所有可调参数（都可以在 .env 里覆盖）
  loader.py        第 1 步  加载文档（.txt / .md / .pdf）
  chunker.py       第 2 步  按句子切块，带重叠
  embedder.py      第 3 步  向量化：手写 TF-IDF  /  BGE 语义模型
  vector_store.py  第 4 步  numpy 矩阵 + 余弦相似度检索
  generator.py     第 5 步  拼 Prompt，流式调用 Claude；多轮追问改写
  pipeline.py      串起整个流程，管理多轮对话
ingest.py          建索引
chat.py            命令行聊天
app.py + static/   网页聊天（可上传文档，可查看检索片段和完整 Prompt）
learn_rag.py       逐步演示
evaluate.py        检索效果评测（测试题在 data/eval_questions.json）
data/docs/         知识库文档（示例是虚构的"星河咖啡"公司资料）
```

## 常用参数（写在 .env 里；生成相关的几项也能在网页设置里改）

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 200 / 40 | 每块最多多少字、相邻块重叠多少字（改了要重跑 `ingest.py`） |
| `EMBEDDER` | `bge` | `bge` 语义向量，`tfidf` 关键词向量（改了要重跑 `ingest.py`） |
| `TOP_K` | 4 | 每次检索取几块给模型 |
| `MIN_SCORE_BGE` | 0.35 | 相似度低于这个值的块丢掉，避免无关问题也硬塞资料 |
| `QUERY_REWRITE` | 1 | 多轮对话时先把"它呢？"这类追问改写成完整问题再检索 |
| `CLAUDE_MODEL` / `EFFORT` | `claude-opus-5-5` / `low` | 生成模型和思考深度 |
| `USE_FALLBACKS` | 1 | 主模型因安全策略拒答时，由 API 自动换模型重答；用代理/中转访问报错时设为 0 |
| `ANTHROPIC_BASE_URL` | 官方地址 | 代理或中转服务的地址 |

## 版本

**v1.0.0**：首个版本。

- 手写 RAG 全流程：加载、切块、TF-IDF / BGE 向量化、余弦相似度检索、拼 Prompt；
- Claude 流式生成，回答带引用编号；多轮对话时自动改写追问；
- 网页版：在网页上配置 API Key 和模型，上传文档、重建索引，查看检索片段和完整 Prompt；
- 命令行聊天、逐步演示脚本、检索评测（Hit@K / MRR）。
