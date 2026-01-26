# app/api/server.py
import asyncio
import json
import os
import re
import shutil
import tempfile
import time
import uuid
from typing import Dict, List, Optional, Literal, Tuple

import anyio
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import OPENAI_API_KEY, LLM_MODEL
from app.core.router import decide_route
from app.tools.web_search import tavily_search

app = FastAPI(title="Multi-Source Research Assistant API", version="0.1.0")

SESSIONS: Dict[str, Dict[str, str]] = {}
PERSIST_COLLECTION = "session_docs"

SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


class SessionResponse(BaseModel):
    session_id: str


class AskRequest(BaseModel):
    question: str


class SourceItem(BaseModel):
    kind: Literal["doc", "web"]
    ref: int
    title: str

    filename: Optional[str] = None
    page: Optional[int] = None
    url: Optional[str] = None


class AskResponse(BaseModel):
    answer: str
    route: str
    routing_reason: str
    sources: List[SourceItem]


# -----------------------------
# Helpers
# -----------------------------
def validate_citations(answer: str, label_map: Dict[str, SourceItem], require_citations_every_bullet: bool):
    used = re.findall(r"\[(\d+)\]", answer)
    used_set = set(used)

    missing = [c for c in used_set if c not in label_map]
    if missing:
        raise HTTPException(status_code=500, detail=f"Model used unknown citations: {missing}")

    if require_citations_every_bullet:
        bullets = [line.strip() for line in answer.splitlines() if line.strip().startswith("•")]
        for i, b in enumerate(bullets, start=1):
            if not re.search(r"\[\d+\]", b):
                raise HTTPException(status_code=500, detail=f"Bullet {i} missing citation")


def renumber_citations(answer: str) -> Tuple[str, Dict[str, int]]:
    found = re.findall(r"\[(\d+)\]", answer)
    unique: List[str] = []
    for f in found:
        if f not in unique:
            unique.append(f)

    mapping: Dict[str, int] = {old: i + 1 for i, old in enumerate(unique)}
    new_answer = answer
    for old, new in mapping.items():
        new_answer = re.sub(rf"\[{old}\]", f"[{new}]", new_answer)

    new_answer = re.sub(r"\n\s*\n+", "\n", new_answer).strip()
    return new_answer, mapping


def apply_renumber_to_sources(used_sources: List[SourceItem], old_to_new: Dict[str, int]) -> List[SourceItem]:
    out: List[SourceItem] = []
    for s in used_sources:
        new_ref = old_to_new.get(str(s.ref))
        if new_ref is None:
            continue
        s.ref = new_ref
        out.append(s)
    out.sort(key=lambda x: x.ref)
    return out


def rewrite_web_query(user_query: str) -> str:
    q = user_query.lower()
    if any(w in q for w in ["fatigue", "eeg", "eog", "perclos", "driver", "drowsiness"]):
        return user_query
    if any(w in q for w in ["methodology", "approach", "technique", "method"]):
        return user_query + " driver fatigue detection EEG EOG deep learning review 2024 2025"
    return user_query


def rewrite_doc_query_for_sections(query: str) -> str:
    q = query.lower()
    section_words = ["conclusion", "limitations", "limitation", "future work", "future scope", "discussion", "summary"]
    if any(w in q for w in section_words):
        return query + " conclusion limitations future work future scope discussion summary findings"
    return query


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)


def get_llm(streaming: bool = False) -> ChatOpenAI:
    return ChatOpenAI(model=LLM_MODEL, temperature=0, api_key=OPENAI_API_KEY, streaming=streaming)


def build_session_db_from_pdf(pdf_path: str, original_filename: str, persist_dir: str) -> int:
    embeddings = get_embeddings()
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    for d in pages:
        d.metadata["source"] = original_filename

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(pages)

    _ = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name=PERSIST_COLLECTION,
    )
    return len(chunks)


def load_session_db(persist_dir: str) -> Chroma:
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name=PERSIST_COLLECTION,
    )


def format_docs_for_prompt(docs, start_index: int):
    parts: List[str] = []
    label_map: Dict[str, SourceItem] = {}
    idx = start_index

    for d in docs:
        filename = d.metadata.get("source") or "uploaded.pdf"
        page = d.metadata.get("page")

        page_human: Optional[int] = None
        if isinstance(page, int):
            page_human = page + 1

        title = f"{filename} — p.{page_human}" if page_human is not None else filename

        label_map[str(idx)] = SourceItem(
            kind="doc",
            ref=idx,
            title=title,
            filename=filename,
            page=page_human,
            url=None,
        )

        parts.append(f"[{idx}] {d.page_content}")
        idx += 1

    return "\n\n".join(parts), label_map, idx


def format_web_for_prompt(results, start_index: int):
    parts: List[str] = []
    label_map: Dict[str, SourceItem] = {}
    idx = start_index

    for r in results:
        title = r.get("title", "Untitled")
        url = r.get("url", "")

        label_map[str(idx)] = SourceItem(
            kind="web",
            ref=idx,
            title=title if title else "Untitled",
            filename=None,
            page=None,
            url=url or None,
        )

        parts.append(f"[{idx}] {r.get('content')}")
        idx += 1

    return "\n\n".join(parts), label_map, idx


def extract_used_numeric_refs(answer_text: str) -> List[int]:
    nums = re.findall(r"\[(\d+)\]", answer_text)
    return [int(n) for n in nums]


def dedupe_sources_in_order(refs: List[int], label_map: Dict[str, SourceItem]) -> List[SourceItem]:
    out: List[SourceItem] = []
    seen = set()

    for ref in refs:
        item = label_map.get(str(ref))
        if not item:
            continue

        if item.kind == "doc":
            key = (item.kind, item.filename, item.page)
        else:
            key = (item.kind, item.url, item.title)

        if key in seen:
            continue

        seen.add(key)
        out.append(item)

    return out


def is_summary_request(q: str) -> bool:
    ql = q.lower()
    return any(t in ql for t in ["summarise", "summarize", "summary", "key points", "bullet points", "in points"])


def is_fixed_count_request(q: str) -> bool:
    ql = q.lower()
    return any(t in ql for t in ["5 points", "five points", "in 5", "exactly 5"])


def build_prompt(
    question: str,
    active_pdf: str,
    decision_route: str,
    decision_reason: str,
    docs_context: str,
    web_context: str,
) -> str:
    if is_fixed_count_request(question):
        instructions = """
Rules:
- Use ONLY the information provided in the context below.
- Output EXACTLY 5 bullet points.
- Each bullet must start with "• ".
- Each bullet must contain ONE clear idea.
- Each bullet MUST end with at least one citation like [1].
- Do NOT invent citations.
"""
        answer_header = "Answer:"

    elif is_summary_request(question):
        instructions = """
Rules:
- Use ONLY the information provided in the context below.
- Output bullet points where EACH bullet is on its OWN LINE.
- Use AS MANY bullet points AS NEEDED (do NOT restrict to 5).
- Each bullet must start with "• " (bullet + space).
- Each bullet MUST end with at least one citation like [1].
- IMPORTANT: Put a newline after EVERY bullet. Do NOT put two bullets on the same line.
- Do NOT write bullets in a paragraph.
- Do NOT invent citations.
"""
        answer_header = "Answer:"

    else:
        instructions = """
Rules:
- Use ONLY the information provided in the context below.
- Write a DETAILED, well-structured explanation in paragraphs.
- Use clear section headings appropriate to the question.
- Use bullet points ONLY if they improve clarity (not required).
- Every paragraph or bullet MUST include at least one citation like [1].
- Do NOT invent citations.
- If the question is NOT about the uploaded PDF and the route is WEB, answer from web sources normally.
"""
        answer_header = "Detailed answer:"

    return f"""
You are a careful research assistant.
{instructions}

Routing decision:
ROUTE: {decision_route}
REASON: {decision_reason}

ACTIVE PDF (if any):
{active_pdf}

CONTEXT:
{docs_context}

{web_context}

Question:
{question}

{answer_header}
""".strip()


def _sse(event: str | None = None, data: dict | str | None = None) -> bytes:
    lines: List[str] = []
    if event:
        lines.append(f"event: {event}")

    if data is None:
        payload = ""
    elif isinstance(data, str):
        payload = data
    else:
        payload = json.dumps(data, ensure_ascii=False)

    for part in str(payload).splitlines() or [""]:
        lines.append(f"data: {part}")

    lines.append("")
    return ("\n".join(lines) + "\n").encode("utf-8")




def _extract_token(chunk) -> Optional[str]:
    """
    LangChain streaming chunk shapes vary by version:
    - AIMessageChunk: chunk.content
    - ChatGenerationChunk: chunk.text or chunk.message.content
    - Sometimes plain str
    """
    if chunk is None:
        return None
    if isinstance(chunk, str):
        return chunk

    c = getattr(chunk, "content", None)
    if isinstance(c, str) and c:
        return c

    t = getattr(chunk, "text", None)
    if isinstance(t, str) and t:
        return t

    msg = getattr(chunk, "message", None)
    if msg is not None:
        mc = getattr(msg, "content", None)
        if isinstance(mc, str) and mc:
            return mc

    return None


# -----------------------------
# API
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/session", response_model=SessionResponse)
def create_session():
    session_id = str(uuid.uuid4())
    persist_dir = tempfile.mkdtemp(prefix="chroma_session_")
    SESSIONS[session_id] = {"persist_dir": persist_dir, "active_pdf_name": ""}
    return {"session_id": session_id}


@app.post("/session/{session_id}/upload")
async def upload_pdf(session_id: str, file: UploadFile = File(...)):
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")

    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    persist_dir = SESSIONS[session_id]["persist_dir"]

    shutil.rmtree(persist_dir, ignore_errors=True)
    os.makedirs(persist_dir, exist_ok=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp_path = tmp.name
        content = await file.read()
        tmp.write(content)

    try:
        chunk_count = build_session_db_from_pdf(tmp_path, file.filename, persist_dir)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    SESSIONS[session_id]["active_pdf_name"] = file.filename
    return {"ok": True, "pdf": file.filename, "num_chunks": chunk_count}


# -----------------------------
# Non-streaming endpoint
# -----------------------------
@app.post("/session/{session_id}/ask", response_model=AskResponse)
def ask(session_id: str, req: AskRequest):
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")

    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    persist_dir = SESSIONS[session_id]["persist_dir"]
    active_pdf = SESSIONS[session_id].get("active_pdf_name", "")

    llm = get_llm(streaming=False)
    decision = decide_route(question)

    if active_pdf and any(x in question.lower() for x in ["this study", "this paper", "this research", "compare"]):
        if decision.route == "WEB":
            decision.route = "BOTH"
            decision.reason += " | Forced DOCS because user refers to uploaded study"

    docs = []
    web_results = []

    if decision.route in ("DOCS", "BOTH"):
        db = load_session_db(persist_dir)
        doc_query = rewrite_doc_query_for_sections(question)
        docs = db.max_marginal_relevance_search(doc_query, k=6, fetch_k=25)

        filtered = []
        for d in docs:
            text = (d.page_content or "").lower()
            if "table of contents" in text:
                continue
            if "appendix" in text and len(text) < 800:
                continue
            filtered.append(d)
        docs = filtered[:6]

    if decision.route in ("WEB", "BOTH"):
        web_query = rewrite_web_query(question)
        web_results = tavily_search(web_query, max_results=5)

    next_idx = 1
    docs_context, doc_map, next_idx = format_docs_for_prompt(docs, start_index=next_idx) if docs else ("", {}, next_idx)
    web_context, web_map, next_idx = format_web_for_prompt(web_results, start_index=next_idx) if web_results else ("", {}, next_idx)

    label_map: Dict[str, SourceItem] = {}
    label_map.update(doc_map)
    label_map.update(web_map)

    prompt = build_prompt(
        question=question,
        active_pdf=active_pdf,
        decision_route=decision.route,
        decision_reason=decision.reason,
        docs_context=docs_context,
        web_context=web_context,
    )

    response = llm.invoke(prompt).content.strip()

    validate_citations(
        response,
        label_map,
        require_citations_every_bullet=is_summary_request(question) or is_fixed_count_request(question),
    )

    old_to_new: Dict[str, int] = {}
    if is_summary_request(question) or is_fixed_count_request(question):
        response, old_to_new = renumber_citations(response)

    if old_to_new:
        old_refs = [int(k) for k in old_to_new.keys()]
        used_sources_raw = dedupe_sources_in_order(old_refs, label_map)
        used_sources = apply_renumber_to_sources(used_sources_raw, old_to_new)
    else:
        used_refs_in_order = extract_used_numeric_refs(response)
        used_sources = dedupe_sources_in_order(used_refs_in_order, label_map)

    return AskResponse(
        answer=response,
        route=decision.route,
        routing_reason=decision.reason,
        sources=used_sources,
    )


# -----------------------------
# Streaming endpoint (SSE)
# -----------------------------
@app.post("/session/{session_id}/ask_stream")
async def ask_stream(session_id: str, req: AskRequest, request: Request):
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")

    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    trace_id = str(uuid.uuid4())
    t0 = time.perf_counter()

    persist_dir = SESSIONS[session_id]["persist_dir"]
    active_pdf = SESSIONS[session_id].get("active_pdf_name", "")

    decision = decide_route(question)

    if active_pdf and any(x in question.lower() for x in ["this study", "this paper", "this research", "compare"]):
        if decision.route == "WEB":
            decision.route = "BOTH"
            decision.reason += " | Forced DOCS because user refers to uploaded study"

    t_retrieval_start = time.perf_counter()
    docs = []
    web_results = []

    if decision.route in ("DOCS", "BOTH"):
        db = load_session_db(persist_dir)
        doc_query = rewrite_doc_query_for_sections(question)
        docs = db.max_marginal_relevance_search(doc_query, k=6, fetch_k=25)

        filtered = []
        for d in docs:
            text = (d.page_content or "").lower()
            if "table of contents" in text:
                continue
            if "appendix" in text and len(text) < 800:
                continue
            filtered.append(d)
        docs = filtered[:6]

    if decision.route in ("WEB", "BOTH"):
        web_query = rewrite_web_query(question)
        web_results = tavily_search(web_query, max_results=5)

    next_idx = 1
    docs_context, doc_map, next_idx = format_docs_for_prompt(docs, start_index=next_idx) if docs else ("", {}, next_idx)
    web_context, web_map, next_idx = format_web_for_prompt(web_results, start_index=next_idx) if web_results else ("", {}, next_idx)

    label_map: Dict[str, SourceItem] = {}
    label_map.update(doc_map)
    label_map.update(web_map)

    t_retrieval = time.perf_counter() - t_retrieval_start

    prompt = build_prompt(
        question=question,
        active_pdf=active_pdf,
        decision_route=decision.route,
        decision_reason=decision.reason,
        docs_context=docs_context,
        web_context=web_context,
    )

    llm = get_llm(streaming=True)

    send, recv = anyio.create_memory_object_stream[bytes](max_buffer_size=500)
    heartbeat_seconds = 10.0

    async def heartbeat():
        try:
            while True:
                await asyncio.sleep(heartbeat_seconds)
                await send.send(b": ping\n\n")
        except asyncio.CancelledError:
            raise
        except Exception:
            return

    async def producer():
        t_gen_start = time.perf_counter()
        full_answer = ""

        try:
            await send.send(_sse("ready", {"trace_id": trace_id}))

            async for chunk in llm.astream(prompt):
                token = _extract_token(chunk)
                if not token:
                   continue

# ✅ Do NOT modify spacing; streamed tokens already include correct whitespace/subwords
                full_answer += token
                await send.send(_sse("token", token))
 

                if await request.is_disconnected():
                    raise asyncio.CancelledError()

            # ✅ Guard: if no tokens were produced, fail loudly
            if not full_answer.strip():
                raise HTTPException(status_code=500, detail="LLM stream produced no tokens (empty answer).")

            validate_citations(
                full_answer,
                label_map,
                require_citations_every_bullet=is_summary_request(question) or is_fixed_count_request(question),
            )

            old_to_new: Dict[str, int] = {}
            renumbered_answer: Optional[str] = None

            if is_summary_request(question) or is_fixed_count_request(question):
                renumbered_answer, old_to_new = renumber_citations(full_answer)
                old_refs = [int(k) for k in old_to_new.keys()]
                used_sources_raw = dedupe_sources_in_order(old_refs, label_map)
                used_sources = apply_renumber_to_sources(used_sources_raw, old_to_new)
            else:
                used_refs_in_order = extract_used_numeric_refs(full_answer)
                used_sources = dedupe_sources_in_order(used_refs_in_order, label_map)

            t_generation = time.perf_counter() - t_gen_start
            t_total = time.perf_counter() - t0

            meta = {
                "trace_id": trace_id,
                "route": decision.route,
                "routing_reason": decision.reason,
                "timings": {
                    "retrieval_s": round(t_retrieval, 4),
                    "generation_s": round(t_generation, 4),
                    "total_s": round(t_total, 4),
                },
                "sources": [s.model_dump() for s in used_sources],
            }

            if renumbered_answer:
                meta["renumbered_answer"] = renumbered_answer

            meta["answer"] = renumbered_answer if renumbered_answer else full_answer

            await send.send(_sse("meta", meta))
            await send.send(_sse("done", {"ok": True}))

        except asyncio.CancelledError:
            raise
        except HTTPException as e:
            await send.send(_sse("error", {"trace_id": trace_id, "error": e.detail}))
        except Exception as e:
            await send.send(_sse("error", {"trace_id": trace_id, "error": str(e)}))
        finally:
            await send.aclose()

    async def event_iterator():
        async with anyio.create_task_group() as tg:
            tg.start_soon(heartbeat)
            tg.start_soon(producer)

            try:
                async for msg in recv:
                    if await request.is_disconnected():
                        tg.cancel_scope.cancel()
                        break
                    yield msg
            finally:
                tg.cancel_scope.cancel()
                await recv.aclose()

    return StreamingResponse(
        event_iterator(),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )
