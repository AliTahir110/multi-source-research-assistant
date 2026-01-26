import chainlit as cl
import re
import tempfile
import shutil
from typing import Optional

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter



from app.core.config import OPENAI_API_KEY, LLM_MODEL
from app.core.router import decide_route
from app.tools.web_search import tavily_search



# --------- Helpers: formatting + web query rewrite ----------

def rewrite_doc_query_for_sections(query: str) -> str:
    q = query.lower()
    if "conclusion" in q or "limitations" in q or "future" in q:
        return (
            query
            + " conclusion limitations future work discussion summary findings"
        )
    return query


def format_docs(docs):
    parts = []
    doc_map = {}
    for i, d in enumerate(docs, start=1):
        label = f"DOC-{i}"
        src = d.metadata.get("source")
        page = d.metadata.get("page")
        doc_map[label] = f"{src} (page {page})"
        parts.append(f"[{label}] {d.page_content}")
    return "\n\n".join(parts), doc_map


def format_web(results):
    parts = []
    web_map = {}
    for i, r in enumerate(results, start=1):
        label = f"WEB-{i}"
        title = r.get("title", "Untitled")
        url = r.get("url", "")
        web_map[label] = f"{title} — {url}"
        parts.append(f"[{label}] {r.get('content')}")
    return "\n\n".join(parts), web_map


def rewrite_web_query(user_query: str) -> str:
    q = user_query.lower()
    if any(w in q for w in ["fatigue", "eeg", "eog", "perclos", "driver", "drowsiness"]):
        return user_query
    if any(w in q for w in ["methodology", "approach", "technique", "method"]):
        return user_query + " driver fatigue detection EEG EOG deep learning review 2024 2025"
    return user_query


# --------- Session PDF ingestion ----------

def build_session_db_from_pdf(
    pdf_path: str,
    original_filename: str,
    embeddings: OpenAIEmbeddings
) -> Chroma:
    """
    Build a session-scoped Chroma DB from a single uploaded PDF.
    Stored in a temp directory so it doesn't mix with your local DB.
    """
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()  # list[Document]

    # overwrite metadata source to keep it clean (filename, not full path)
    for d in pages:
        d.metadata["source"] = original_filename

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(pages)

    # Persist in a unique temp folder for this session upload
    persist_dir = tempfile.mkdtemp(prefix="chroma_session_")

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name="session_docs",
    )

    # Keep track of where it lives so we can clean it later if needed
    cl.user_session.set("persist_dir", persist_dir)
    cl.user_session.set("active_pdf_name", original_filename)

    return db


async def ask_and_load_pdf(embeddings: OpenAIEmbeddings) -> Optional[Chroma]:
    """
    Prompts user to upload a PDF and builds a session DB.
    Returns None if user cancels.
    """
    files = await cl.AskFileMessage(
        content="📤 Upload a PDF (session-only). I will answer ONLY from this PDF + web if needed.",
        accept=["application/pdf"],
        max_size_mb=25,
        timeout=300,
    ).send()

    if not files:
        return None

    f = files[0]
    pdf_path = f.path
    pdf_name = f.name

    # Clean up previous session DB folder if any
    old_dir = cl.user_session.get("persist_dir")
    if old_dir:
        try:
            shutil.rmtree(old_dir, ignore_errors=True)
        except Exception:
            pass

    db = build_session_db_from_pdf(pdf_path, pdf_name, embeddings)
    return db


# --------- Chainlit lifecycle ----------
def rewrite_doc_query_for_sections(query: str) -> str:
    """
    If user asks about structural sections (conclusion/limitations/future work),
    expand the query with section keywords to retrieve those chunks more reliably.
    """
    q = query.lower()
    section_words = ["conclusion", "limitations", "limitation", "future work", "future scope", "discussion", "summary"]
    if any(w in q for w in section_words):
        return query + " conclusion limitations future work future scope discussion summary findings"
    return query


@cl.on_chat_start
async def on_chat_start():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY,
    )

    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0,
        api_key=OPENAI_API_KEY,
    )

    cl.user_session.set("embeddings", embeddings)
    cl.user_session.set("llm", llm)

    await cl.Message(
        content=(
            "📚 Multi-Source Research Assistant (Session PDF Mode)\n\n"
            "- Upload a PDF and ask questions about it.\n"
            "- Type **upload** or **/upload** anytime to replace the PDF.\n"
            "- I can also use the web for up-to-date comparisons.\n"
        )
    ).send()

    db = await ask_and_load_pdf(embeddings)
    if db is None:
        await cl.Message(content="No PDF uploaded yet. Type **upload** when you're ready.").send()
        cl.user_session.set("db", None)
        return

    cl.user_session.set("db", db)
    pdf_name = cl.user_session.get("active_pdf_name", "your PDF")
    await cl.Message(content=f"✅ Loaded **{pdf_name}**. Ask me anything about it.").send()


@cl.on_message
async def on_message(message: cl.Message):
    llm = cl.user_session.get("llm")
    embeddings = cl.user_session.get("embeddings")
    db = cl.user_session.get("db")

    query = message.content.strip()

    # Allow replacing the PDF anytime
    if query.lower() in {"upload", "/upload", "change pdf", "new pdf"}:
        new_db = await ask_and_load_pdf(embeddings)
        if new_db is None:
            await cl.Message(content="Upload cancelled. Current PDF remains active.").send()
            return
        cl.user_session.set("db", new_db)
        pdf_name = cl.user_session.get("active_pdf_name", "your PDF")
        await cl.Message(content=f"✅ Switched to **{pdf_name}**. Ask your question now.").send()
        return

    if db is None:
        await cl.Message(content="Please upload a PDF first. Type **upload**.").send()
        return

    # 1) Decide route
    decision = decide_route(query)

    # 2) Gather context based on route
    docs = []
    web_results = []

    if decision.route in ("DOCS", "BOTH"):
        doc_query = rewrite_doc_query_for_sections(query)
        docs = db.max_marginal_relevance_search(doc_query, k=6, fetch_k=25)


        # Filter out TOC/appendix-like chunks
        filtered = []
        for d in docs:
            text = (d.page_content or "").lower()
            if "table of contents" in text:
                continue
            if "appendix" in text and len(text) < 800:
                continue
            filtered.append(d)
        docs = filtered[:4]

    if decision.route in ("WEB", "BOTH"):
        web_query = rewrite_web_query(query)
        web_results = tavily_search(web_query, max_results=5)

    docs_context, doc_citations = format_docs(docs) if docs else ("", {})
    web_context, web_citations = format_web(web_results) if web_results else ("", {})

    # 3) Prompt with strict inline citations
    prompt = f"""
You are a careful research assistant.

Rules:
- Use ONLY the information provided in the context below.
- EVERY sentence must end with at least one citation tag like [DOC-1] or [WEB-2].
- Do NOT invent citations.
- If the answer cannot be found, say: "I don't know based on the available sources."

Routing decision:
ROUTE: {decision.route}
REASON: {decision.reason}

DOCUMENT CONTEXT:
{docs_context}

WEB CONTEXT:
{web_context}

Question:
{query}

Answer (with inline citations):
"""

    response = llm.invoke(prompt).content.strip()

    # 4) Legend: only sources actually cited
    all_citations = {**doc_citations, **web_citations}
    used_labels = set(re.findall(r"\[(DOC-\d+|WEB-\d+)\]", response))

    legend = ""
    if used_labels:
        legend += "\n\n---\n**Sources**\n"
        for label in sorted(used_labels, key=lambda x: (x.split("-")[0], int(x.split("-")[1]))):
            ref = all_citations.get(label)
            if ref:
                legend += f"- [{label}] {ref}\n"

    footer = (
        f"\n\n---\n**Route Used:** {decision.route}\n"
        f"**Routing Reason:** {decision.reason}"
    )

    await cl.Message(content=response + footer + legend).send()
