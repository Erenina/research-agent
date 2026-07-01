"""
FastAPI web arayüzü — araştırma agent'ını tarayıcıdan çalıştır ve ReAct izini
CANLI göster (Server-Sent Events / SSE ile).

Neden SSE? Agent birkaç adım sürer (ara → oku → hesapla → cevap). Cevabı sonuna
kadar bekletmek yerine, her adım oluştuğunda tarayıcıya anında gönderiyoruz;
kullanıcı agent'ı "düşünürken" izliyor. run_agent'ın on_event callback'i, adımları
bir kuyruğa (queue) yazıyor; SSE üreteci de kuyruktan okuyup akıtıyor.

Uç noktalar:
  GET /            -> tek sayfa arayüz
  GET /health      -> sağlık kontrolü
  GET /ask?q=...   -> SSE akışı (agent düşündükçe adımları gönderir)
"""

import json
import queue
import threading
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from agent.core import run_agent, AgentError

app = FastAPI(title="Research Agent")

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    """Tek sayfa arayüzü servis et."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


def _event_payload(ev: tuple) -> dict:
    """core.py'nin (tuple) olayını, arayüzün anlayacağı JSON'a çevir."""
    kind = ev[0]
    if kind == "think":
        return {"type": "think", "text": ev[1]}
    if kind == "tool_call":
        return {"type": "tool_call", "name": ev[1], "args": ev[2]}
    if kind == "tool_result":
        return {"type": "tool_result", "name": ev[1], "result": str(ev[2])}
    if kind == "final":
        return {"type": "final", "answer": ev[1]}
    if kind == "error":
        return {"type": "error", "message": ev[1]}
    return {"type": "unknown"}


@app.get("/ask")
def ask(q: str):
    """Soruyu çalıştır; her adımı oluştukça SSE olayı olarak gönder."""

    def stream():
        events: queue.Queue = queue.Queue()

        def on_event(*ev):
            events.put(ev)

        def worker():
            # Agent'ı ayrı bir thread'de koştur ki olaylar oluştukça akıtabilelim
            try:
                run_agent(q, on_event=on_event)
            except AgentError as e:
                events.put(("error", str(e)))
            except Exception as e:  # beklenmedik hata arayüzü çökertmesin
                events.put(("error", f"Beklenmedik hata: {e}"))
            finally:
                events.put(None)  # bitiş işareti (sentinel)

        threading.Thread(target=worker, daemon=True).start()

        while True:
            ev = events.get()
            if ev is None:
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                break
            yield f"data: {json.dumps(_event_payload(ev), ensure_ascii=False)}\n\n"

    # SSE için doğru media type + tampon kapatma başlıkları
    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
