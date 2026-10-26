"""全局配置。

所有参数都可以在项目根目录的 .env 文件里覆盖（格式：KEY=VALUE），.env 优先于系统环境变量。
改完参数后记得重新运行 `python ingest.py` 重建索引（切块/向量化相关的参数才需要）。
API Key、模型等生成相关的参数也可以在网页的「设置」里改，保存后同样写进 .env。
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"


def _load_dotenv(path: Path) -> None:
    """极简版 .env 读取。"""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip().strip('"').strip("'")


_load_dotenv(ENV_FILE)

# ---------- 知识库 ----------
DOCS_DIR = ROOT / os.getenv("DOCS_DIR", "data/docs")   # 放原始文档的目录（.txt / .md / .pdf）
INDEX_DIR = ROOT / os.getenv("INDEX_DIR", "data/index")  # 建好的向量索引存这里

# ---------- 第 2 步：切块 ----------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "200"))       # 每块最多多少个字符
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "40"))  # 相邻两块重叠多少个字符

# ---------- 第 3 步：向量化 ----------
# bge   = 神经网络语义向量（BAAI/bge-small-zh-v1.5，首次运行自动下载约 95MB）
# tfidf = 纯 numpy 手写的关键词向量，不需要下载任何模型，用来对比理解
EMBEDDER = os.getenv("EMBEDDER", "bge")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-zh-v1.5")

# ---------- 第 4 步：检索 ----------
TOP_K = int(os.getenv("TOP_K", "4"))                # 每次取最相似的几块
# 相似度低于这个值的块直接丢弃（防止无关问题也硬塞资料）。
# 两种向量的分数范围差别很大：bge 相关内容一般 0.4~0.7，tfidf 一般 0.1~0.3，所以分开设
MIN_SCORE = {
    "bge": float(os.getenv("MIN_SCORE_BGE", "0.35")),
    "tfidf": float(os.getenv("MIN_SCORE_TFIDF", "0.05")),
}
QUERY_REWRITE = os.getenv("QUERY_REWRITE", "1") == "1"  # 多轮对话时先把追问改写成完整问题再检索

# ---------- 第 5 步：生成 ----------
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "16000"))
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "6"))  # 多轮对话最多保留几轮

# 下面这几项可以在网页「设置」里修改
API_KEY = BASE_URL = CLAUDE_MODEL = EFFORT = ""
USE_FALLBACKS = True
SETTING_KEYS = ("ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "CLAUDE_MODEL", "EFFORT", "USE_FALLBACKS")


def _apply_settings() -> None:
    global API_KEY, BASE_URL, CLAUDE_MODEL, EFFORT, USE_FALLBACKS
    API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    BASE_URL = os.getenv("ANTHROPIC_BASE_URL", "")      # 留空 = 官方地址；用代理/中转时填它的地址
    CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-5-5")
    EFFORT = os.getenv("EFFORT", "low")                 # low / medium / high：越高越仔细，也越慢越贵
    # 服务端兜底：主模型因安全策略拒答时，API 自动换一个模型重答。代理/中转不支持时设为 0
    USE_FALLBACKS = os.getenv("USE_FALLBACKS", "1") == "1"


_apply_settings()


def save_settings(values: dict[str, str]) -> None:
    """把设置写进 .env（重启后仍有效）并立即生效。值为空字符串表示删掉这一项、恢复默认。"""
    lines = ENV_FILE.read_text(encoding="utf-8").splitlines() if ENV_FILE.exists() else []
    pending = dict(values)
    output = []
    for line in lines:
        key = line.split("=", 1)[0].strip()
        if "=" in line and not line.lstrip().startswith("#") and key in pending:
            value = pending.pop(key)
            if value:
                output.append(f"{key}={value}")
            continue
        output.append(line)
    output += [f"{key}={value}" for key, value in pending.items() if value]
    ENV_FILE.write_text("\n".join(output) + "\n", encoding="utf-8")

    for key, value in values.items():
        if value:
            os.environ[key] = value
        else:
            os.environ.pop(key, None)
    _apply_settings()
