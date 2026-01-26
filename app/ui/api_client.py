# app/ui/api_client.py
from __future__ import annotations

import json
import os
from typing import Any, AsyncIterator, Optional, Tuple

import httpx


class APIClient:
    """
    SSE client for FastAPI backend.

    Key stability:
    - read=None (no streaming read timeout)
    - robust SSE parsing
    - DO NOT strip token whitespace (spaces are meaningful)
    """

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = (base_url or os.getenv("API_BASE_URL") or "http://127.0.0.1:8001").rstrip("/")

    async def create_session(self) -> str:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(f"{self.base_url}/session")
            r.raise_for_status()
            return r.json()["session_id"]

    async def upload_pdf(self, session_id: str, file_name: str, file_bytes: bytes) -> dict:
        files = {"file": (file_name, file_bytes, "application/pdf")}
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(f"{self.base_url}/session/{session_id}/upload", files=files)
            r.raise_for_status()
            return r.json()

    async def ask(self, session_id: str, question: str) -> dict:
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(f"{self.base_url}/session/{session_id}/ask", json={"question": question})
            r.raise_for_status()
            return r.json()

    async def ask_stream(self, session_id: str, question: str) -> AsyncIterator[Tuple[str, Any]]:
        url = f"{self.base_url}/session/{session_id}/ask_stream"

        timeout = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=None)
        limits = httpx.Limits(max_connections=20, max_keepalive_connections=20, keepalive_expiry=60.0)
        transport = httpx.AsyncHTTPTransport(retries=0)

        async with httpx.AsyncClient(timeout=timeout, limits=limits, transport=transport) as client:
            async with client.stream(
                "POST",
                url,
                json={"question": question},
                headers={"Accept": "text/event-stream"},
            ) as r:
                r.raise_for_status()

                event: Optional[str] = None
                data_lines: list[str] = []

                async for line in r.aiter_lines():
                    # end of message
                    if line == "":
                        if event is None and not data_lines:
                            continue

                        raw = "\n".join(data_lines)
                        data_lines.clear()

                        ev = event or "message"
                        event = None

                        if ev == "token":
                            # ✅ token is plain text; keep whitespace
                            yield "token", raw

                        elif ev in ("meta", "error", "ready", "done"):
                            try:
                                yield ev, json.loads(raw) if raw.strip() else {}
                            except Exception:
                                yield ev, raw
                        else:
                            yield ev, raw

                        continue

                    # heartbeat
                    if line.startswith(":"):
                        continue

                    if line.startswith("event:"):
                        event = line[len("event:") :].strip()
                        continue

                    if line.startswith("data:"):
                        # ✅ IMPORTANT: only remove ONE optional leading space after "data:"
                        payload = line[len("data:") :]
                        if payload.startswith(" "):
                            payload = payload[1:]
                        data_lines.append(payload)
                        continue

                    data_lines.append(line)
