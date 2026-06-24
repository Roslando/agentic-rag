"""FastAPI server — serves the vanilla HTML/CSS/JS chat UI."""

from __future__ import annotations
import asyncio
import json
import logging
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

logger = logging.getLogger(__name__)

_STATIC = Path(__file__).parent / "static"
_ASSETS = Path(__file__).parent / "assets"

_rag = None


def _get_rag():
    global _rag
    if _rag is None:
        from src.core.rag_system import RAGSystem
        _rag = RAGSystem()
    return _rag


app = FastAPI(title="Corporate Knowledge Assistant", docs_url=None, redoc_url=None)
app.mount("/assets", StaticFiles(directory=str(_ASSETS)), name="assets")
app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")


class ChatRequest(BaseModel):
    message: str


@app.get("/", response_class=HTMLResponse)
async def index():
    return (_STATIC / "index.html").read_text(encoding="utf-8")


@app.post("/api/chat")
async def chat_stream(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Empty message")

    async def generate():
        try:
            gen = _get_rag().stream_chat(req.message)
            loop = asyncio.get_running_loop()

            def _next():
                return next(gen, None)

            while True:
                chunk = await loop.run_in_executor(None, _next)
                if chunk is None:
                    break
                yield f"data: {json.dumps({'token': chunk})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.exception("Stream error")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/reset")
async def reset_conversation():
    if _rag:
        _rag.reset_conversation()
    return {"ok": True}


@app.get("/api/stats")
async def get_stats():
    try:
        info = _get_rag().get_collection_stats()
        return info
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/upload")
async def upload_pdfs(files: list[UploadFile] = File(...)):
    from src.config import DOCS_DIR, MARKDOWN_DIR
    from src.ingestion.pdf_converter import convert_pdf_to_markdown
    from src.ingestion.chunker import chunk_markdown_document
    from src.ingestion.indexer import save_parents, index_children
    from src.retrieval.vector_store import get_qdrant_client, ensure_collection

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    client = get_qdrant_client()
    ensure_collection(client)

    results = []
    total = 0
    for upload in files:
        content = await upload.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)
        dest = DOCS_DIR / (upload.filename or tmp_path.name)
        shutil.copy(tmp_path, dest)
        tmp_path.unlink(missing_ok=True)
        try:
            md_path = convert_pdf_to_markdown(dest, MARKDOWN_DIR)
            parents, children = chunk_markdown_document(md_path)
            save_parents(parents, md_path.stem)
            n = index_children(client, children, source_name=md_path.stem)
            total += n
            results.append({"file": upload.filename, "status": "ok", "chunks": n})
        except Exception as e:
            results.append({"file": upload.filename, "status": "error", "message": str(e)})

    global _rag
    _rag = None
    return {"results": results, "total": total}


def launch(port: int = 7860) -> None:
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
