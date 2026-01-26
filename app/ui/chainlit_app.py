# app/ui/chainlit_app.py
from __future__ import annotations

import chainlit as cl
from app.ui.api_client import APIClient

api = APIClient()  # uses API_BASE_URL or defaults to http://127.0.0.1:8001


def _format_sources_md(sources: list | None) -> str:
    if not sources:
        return "No sources returned."

    lines = []
    for s in sources:
        if not isinstance(s, dict):
            continue

        ref = s.get("ref")
        title = s.get("title") or "Source"
        url = s.get("url")

        if url:
            lines.append(f"- **[{ref}]** [{title}]({url})")
        else:
            lines.append(f"- **[{ref}]** {title}")

    return "\n".join(lines) if lines else "No sources returned."


async def _ensure_session() -> str:
    session_id = cl.user_session.get("session_id")
    if session_id:
        return session_id

    session_id = await api.create_session()
    cl.user_session.set("session_id", session_id)
    return session_id


async def _ask_and_upload_pdf(session_id: str) -> bool:
    files = await cl.AskFileMessage(
        content="📤 Upload a PDF (I will send it to the backend session).",
        accept=["application/pdf"],
        max_size_mb=25,
        timeout=300,
    ).send()

    if not files:
        return False

    f = files[0]

    status = cl.Message(content=f"⬆️ Uploading **{f.name}** to backend…")
    await status.send()

    try:
        with open(f.path, "rb") as fp:
            pdf_bytes = fp.read()

        result = await api.upload_pdf(session_id=session_id, file_name=f.name, file_bytes=pdf_bytes)
        cl.user_session.set("active_pdf_name", f.name)

        chunks = result.get("num_chunks") or result.get("chunks") or result.get("chunk_count")
        extra_txt = f"\n(chunks: `{chunks}`)" if chunks is not None else ""

        await cl.Message(content=f"✅ Uploaded **{f.name}** successfully.{extra_txt}").send()
        return True

    except Exception as e:
        await cl.Message(content=f"❌ Upload failed: `{e}`").send()
        return False

    finally:
        await status.remove()


@cl.on_chat_start
async def on_chat_start():
    session_id = await _ensure_session()

    await cl.Message(
        content=(
            "📚 **Multi-Source Research Assistant (Thin UI → FastAPI)**\n\n"
            f"✅ Backend session created: `{session_id}`\n\n"
            "**Commands:**\n"
            "- Type **upload** to upload/replace a PDF (sent to backend)\n"
            "- Ask questions normally\n\n"
            "Now supports **streaming answers** + trace + timings.\n"
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    session_id = await _ensure_session()
    text = (message.content or "").strip()

    if text.lower() in {"upload", "/upload", "change pdf", "new pdf"}:
        ok = await _ask_and_upload_pdf(session_id)
        if not ok:
            await cl.Message(content="Upload cancelled.").send()
        return

    if not text:
        return

    streamed_msg = cl.Message(content="")
    await streamed_msg.send()

    meta = None
    streamed_any_token = False

    try:
        async for ev_type, data in api.ask_stream(session_id=session_id, question=text):
            if ev_type == "token":
                if isinstance(data, str) and data:
                    streamed_any_token = True
                await streamed_msg.stream_token(data if isinstance(data, str) else str(data))

            elif ev_type == "meta":
                meta = data

            elif ev_type == "error":
                await cl.Message(content=f"❌ Backend error: `{data}`").send()
                return

        # ✅ FINAL FALLBACK: if tokens didn't show, render final answer from meta
        if meta:
            final_answer = None
            renum = meta.get("renumbered_answer")
            ans = meta.get("answer")  # provided by updated server.py

            if isinstance(renum, str) and renum.strip():
                final_answer = renum.strip()
            elif isinstance(ans, str) and ans.strip():
                final_answer = ans.strip()

            # If token streaming produced nothing, update message with final answer
            if final_answer and (not streamed_any_token or not (streamed_msg.content or "").strip()):
                streamed_msg.content = final_answer
                await streamed_msg.update()

            timings = meta.get("timings", {}) or {}
            route = meta.get("route")
            reason = meta.get("routing_reason")
            trace_id = meta.get("trace_id")
            sources = meta.get("sources", [])

            await cl.Message(
                content=(
                    "---\n"
                    f"**Route:** `{route}`\n"
                    f"**Reason:** {reason}\n"
                    f"**Trace ID:** `{trace_id}`\n"
                    f"**Timings:** retrieval `{timings.get('retrieval_s')}s`, "
                    f"generation `{timings.get('generation_s')}s`, total `{timings.get('total_s')}s`"
                )
            ).send()

            sources_md = _format_sources_md(sources)
            await cl.Message(content=f"**Sources**\n\n{sources_md}").send()

    except Exception as e:
        await cl.Message(content=f"❌ Streaming failed: `{e}`").send()
