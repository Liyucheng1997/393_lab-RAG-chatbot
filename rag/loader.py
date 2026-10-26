"""第 1 步：加载文档。

把各种格式的文件读成纯文本。PDF 按页拆开，这样回答时能告诉你出自第几页。
"""
from dataclasses import dataclass
from pathlib import Path

SUPPORTED = {".txt", ".md", ".pdf"}


@dataclass
class Document:
    source: str          # 文件名，回答里引用来源时用
    text: str
    page: int | None = None  # 只有 PDF 才有页码


def load_file(path: Path) -> list[Document]:
    suffix = path.suffix.lower()
    if suffix in (".txt", ".md"):
        return [Document(source=path.name, text=path.read_text(encoding="utf-8"))]
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(path)
        docs = []
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                docs.append(Document(source=path.name, text=text, page=i))
        return docs
    return []


def load_documents(docs_dir: Path) -> list[Document]:
    docs: list[Document] = []
    for path in sorted(docs_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            docs.extend(load_file(path))
    return docs
