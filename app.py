"""网页版聊天：python app.py，然后打开 http://127.0.0.1:8000"""
import json
import os
import uuid

import anthropic
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag import __version__, config
from rag.loader import SUPPORTED
from rag.pipeline import Conversation, RAGPipeline, build_index
from rag.vector_store import VectorStore

app = FastAPI(title="RAG 聊天机器人")
app.mount("/static", StaticFiles(directory=config.ROOT / "static"), name="static")

state: dict = {"pipeline": None}
conversations: dict[str, Conversation] = {}  # 会话 id → 多轮对话历史（存在内存里，重启即清空）


def get_pipeline() -> RAGPipeline:
    if state["pipeline"] is None:
        if not VectorStore.exists(config.INDEX_DIR):
            build_index()
        state["pipeline"] = RAGPipeline()
    return state["pipeline"]


class ChatRequest(BaseModel):
    question: str
    session_id: str | None = None


@app.get("/")
def index():
    return FileResponse(config.ROOT / "static" / "index.html")


@app.get("/api/stats")
def stats():
    store = get_pipeline().store
    return {
        "chunks": len(store.chunks),
        "sources": store.meta["sources"],
        "embedder": store.meta["embedder"],
        "chunk_size": store.meta["chunk_size"],
        "chunk_overlap": store.meta["chunk_overlap"],
        "built_at": store.meta["built_at"],
        "model": config.CLAUDE_MODEL,
        "top_k": config.TOP_K,
        "version": __version__,
    }


# ---------------------------------------------------------------------------
# 设置：API Key、接口地址、模型等。保存到 .env，完整的 Key 永远不会返回给浏览器
# ---------------------------------------------------------------------------

MODELS = ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5", "claude-fable-5-1"]
EFFORTS = ["low", "medium", "high"]


class SettingsRequest(BaseModel):
    api_key: str | None = None  # None = 不修改；"" = 清除
    base_url: str = ""
    model: str = "claude-opus-5-5"
    effort: str = "low"
    use_fallbacks: bool = True


def settings_view() -> dict:
    key = config.API_KEY
    return {
        "api_key_set": bool(key),
        "api_key_hint": f"{key[:7]}…{key[-4:]}" if len(key) > 16 else ("已设置" if key else ""),
        "base_url": config.BASE_URL,
        "model": config.CLAUDE_MODEL,
        "effort": config.EFFORT,
        "use_fallbacks": config.USE_FALLBACKS,
        "models": MODELS,
        "efforts": EFFORTS,
    }


def validated(req: SettingsRequest) -> tuple[str, str]:
    model = req.model.strip()
    base_url = req.base_url.strip().rstrip("/")
    if not model:
        raise HTTPException(400, "模型名不能为空")
    if req.effort not in EFFORTS:
        raise HTTPException(400, f"思考深度只能是 {' / '.join(EFFORTS)}")
    if base_url and not base_url.startswith(("http://", "https://")):
        raise HTTPException(400, "接口地址要以 http:// 或 https:// 开头")
    return model, base_url


@app.get("/api/settings")
def get_settings():
    return settings_view()


@app.post("/api/settings")
def update_settings(req: SettingsRequest):
    model, base_url = validated(req)
    values = {
        "ANTHROPIC_BASE_URL": base_url,
        "CLAUDE_MODEL": model,
        "EFFORT": req.effort,
        "USE_FALLBACKS": "1" if req.use_fallbacks else "0",
    }
    if req.api_key is not None:
        values["ANTHROPIC_API_KEY"] = req.api_key.strip()
    config.save_settings(values)
    if state["pipeline"] is not None:
        state["pipeline"].reset_generator()
    return settings_view()


@app.post("/api/settings/test")
def test_settings(req: SettingsRequest):
    """用表单里的值（还没保存也行）查询一次模型信息：能验证 Key、地址和模型名，且不消耗 token。"""
    model, base_url = validated(req)
    api_key = (req.api_key if req.api_key is not None else config.API_KEY).strip()
    if not api_key:
        return {"ok": False, "message": "还没有填写 API Key"}
    try:
        client = anthropic.Anthropic(api_key=api_key, base_url=base_url or None, max_retries=0, timeout=15)
        info = client.models.retrieve(model)
        return {"ok": True, "message": f"连接成功：{info.display_name}"}
    except anthropic.AuthenticationError:
        return {"ok": False, "message": "API Key 无效"}
    except anthropic.NotFoundError:
        return {"ok": False, "message": f"找不到模型 {model}（检查模型名；部分中转不支持查询模型，可以直接保存后提问试试）"}
    except anthropic.APIStatusError as e:
        return {"ok": False, "message": f"接口返回错误 {e.status_code}：{e.message}"}
    except anthropic.APIConnectionError:
        return {"ok": False, "message": "连不上接口地址，请检查网络、代理或地址是否正确"}


@app.post("/api/chat")
def chat(req: ChatRequest):
    """以 NDJSON 流返回：每行一个事件（rewrite / sources / prompt / text / error / done）。"""
    pipeline = get_pipeline()
    session_id = req.session_id or uuid.uuid4().hex
    conv = conversations.setdefault(session_id, Conversation())

    def events():
        yield json.dumps({"type": "session", "id": session_id}) + "\n"
        for event in pipeline.ask(req.question, conv):
            yield json.dumps(event, ensure_ascii=False) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")


@app.post("/api/upload")
async def upload(file: UploadFile):
    name = (file.filename or "").replace("\\", "/").split("/")[-1]
    if not name or not any(name.lower().endswith(ext) for ext in SUPPORTED):
        raise HTTPException(400, "只支持 .txt / .md / .pdf 文件")
    config.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (config.DOCS_DIR / name).write_bytes(await file.read())
    return reindex()


@app.post("/api/reindex")
def reindex():
    build_index()
    state["pipeline"] = RAGPipeline()
    conversations.clear()  # 知识库变了，旧对话里的引用编号已经对不上
    return stats()


if __name__ == "__main__":
    get_pipeline()  # 启动时就加载好模型，第一次提问不用等
    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("PORT", "8000")))
