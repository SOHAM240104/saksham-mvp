# """
# server.py — COMMENTED OUT (Streamlit-only)
# ==========================================
# FastAPI backend for Saksham MVP: Evolution API webhook listener.
# Receives WhatsApp messages, runs the RAG agent, and sends replies via Evolution API.
# """

# import os

# # Avoid OpenMP duplicate library crash on macOS (FAISS/numpy/sentence-transformers)
# os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# from typing import Any, Optional

# from dotenv import load_dotenv
# from fastapi import FastAPI, Request, Body
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse, HTMLResponse
# import requests

# from agent_core import get_agent_response, get_last_docs
# from pdf_generator import create_sources_pdf

# load_dotenv()

# app = FastAPI(title="Saksham Care API", version="1.0.0")

# # Allow browser requests from any origin (e.g. localhost, 127.0.0.1, file://)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=False,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# def _evo_config() -> tuple:
#     """Return (base_url, api_key, instance) from env, or empty strings."""
#     base_url = (os.environ.get("EVOLUTION_API_URL") or "").rstrip("/")
#     api_key = os.environ.get("EVOLUTION_API_KEY", "")
#     instance = os.environ.get("EVOLUTION_INSTANCE_NAME", "")
#     return base_url, api_key, instance


# def _evolution_send_text(remote_jid: str, text: str) -> bool:
#     """Send a text message via Evolution API. Returns True on success."""
#     base_url, api_key, instance = _evo_config()
#     if not base_url or not api_key or not instance:
#         print("[EVOLUTION] sendText skipped: missing EVOLUTION_API_URL / KEY / INSTANCE in .env")
#         return False
#     number = remote_jid if "@" in remote_jid else f"{remote_jid}@s.whatsapp.net"
#     url = f"{base_url}/message/sendText/{instance}"
#     headers = {"apikey": api_key}
#     payload = {"number": number, "text": text}
#     try:
#         r = requests.post(url, json=payload, headers=headers, timeout=30)
#         ok = r.status_code in (200, 201)
#         if not ok:
#             print(f"[EVOLUTION] sendText failed: status={r.status_code} body={r.text[:400]}")
#         return ok
#     except Exception as e:
#         print(f"[EVOLUTION] sendText error: {e}")
#         return False


# def _evolution_send_pdf(remote_jid: str, pdf_b64: str) -> bool:
#     """Send a PDF document via Evolution API sendMedia. Returns True on success."""
#     base_url, api_key, instance = _evo_config()
#     if not base_url or not api_key or not instance:
#         print("[EVOLUTION] sendPDF skipped: missing EVOLUTION_API_URL / KEY / INSTANCE in .env")
#         return False
#     number = remote_jid if "@" in remote_jid else f"{remote_jid}@s.whatsapp.net"
#     url = f"{base_url}/message/sendMedia/{instance}"
#     headers = {"apikey": api_key, "Content-Type": "application/json"}
#     payload = {
#         "number": number,
#         "mediatype": "document",
#         "mimetype": "application/pdf",
#         "caption": "Care Support Sources",
#         "media": pdf_b64,
#         "fileName": "Care_Support_Sources.pdf",
#     }
#     try:
#         r = requests.post(url, json=payload, headers=headers, timeout=60)
#         print(f"[EVOLUTION] sendMedia status={r.status_code} body={r.text[:500]}")
#         return r.status_code in (200, 201)
#     except Exception as e:
#         print(f"[EVOLUTION] sendMedia error: {e}")
#         return False


# def _extract_incoming_message(body: dict) -> tuple[Optional[str], Optional[str]]:
#     """
#     Parse Evolution API webhook payload. Returns (remote_jid, text) or (None, None).
#     Only processes incoming messages (fromMe is False).
#     Handles data as single object or list (MESSAGES_UPSERT can send batch).
#     """
#     try:
#         data = body.get("data") or body
#         # Evolution can send data as a list of messages; take first
#         if isinstance(data, list):
#             data = data[0] if data else {}
#         key = data.get("key") or {}
#         if key.get("fromMe") is True:
#             return None, None
#         remote_jid = key.get("remoteJid")
#         if not remote_jid:
#             return None, None
#         # Ignore group chats — only reply in 1:1 DMs
#         if isinstance(remote_jid, str) and "@g.us" in remote_jid:
#             print(f"[WEBHOOK] Ignored group message from {remote_jid}")
#             return None, None
#         # Never reply to status broadcast or non-chat JIDs
#         if isinstance(remote_jid, str) and ("status@broadcast" in remote_jid or "@broadcast" in remote_jid):
#             return None, None
#         message = data.get("message") or {}
#         text = message.get("conversation")
#         if not text and isinstance(message.get("extendedTextMessage"), dict):
#             text = (message.get("extendedTextMessage") or {}).get("text")
#         text = (text or "").strip()
#         return remote_jid, text if text else None
#     except Exception as e:
#         print(f"[WEBHOOK] extract error: {e}")
#         return None, None


# @app.post("/webhook/evolution")
# async def webhook_evolution(request: Request):
#     """
#     Evolution API webhook: receive WhatsApp message, run agent, send reply.
#     Ignores malformed payloads, fromMe messages, and status broadcasts. Replies only to the sender's JID from this request.
#     """
#     try:
#         body = await request.json()
#     except Exception as e:
#         print(f"[WEBHOOK] Invalid JSON: {e}")
#         return JSONResponse(
#             content={"ok": False, "error": "Invalid JSON"},
#             status_code=400,
#         )

#     # Only process new-message events; ignore read receipts, delivery, typing, etc.
#     ev = (body.get("event") or "").strip().lower().replace("_", ".")
#     if ev != "messages.upsert":
#         print(f"[WEBHOOK] Ignoring non-message event: {ev!r}")
#         return JSONResponse(
#             content={"ok": True, "handled": False, "reason": f"Ignored event: {ev}"},
#         )

#     data = body.get("data") or body
#     if isinstance(data, list):
#         data = data[0] if data else {}
#     print(f"[WEBHOOK] event={ev!r} has_key={'key' in data} has_message={'message' in data}")

#     remote_jid, text = _extract_incoming_message(body)
#     if remote_jid is None:
#         return JSONResponse(content={"ok": True, "handled": False})

#     if not text:
#         return JSONResponse(content={"ok": True, "handled": True})

#     print(f"[WEBHOOK] from {remote_jid!r} text={text[:60]!r}...")
#     try:
#         response_text = get_agent_response(remote_jid, text)
#     except Exception as e:
#         print(f"[AGENT] error: {e}")
#         response_text = "Something went wrong. Please try again."

#     if response_text:
#         sent = _evolution_send_text(remote_jid, response_text)
#         print(f"[WEBHOOK] reply sent={sent} to {remote_jid!r}")

#     if response_text and "generating a pdf" in response_text.lower():
#         docs = get_last_docs(remote_jid)
#         if docs:
#             try:
#                 pdf_b64 = create_sources_pdf(docs)
#                 print(f"[PDF] generated {len(pdf_b64)} chars base64 from {len(docs)} docs")
#                 _evolution_send_pdf(remote_jid, pdf_b64)
#             except Exception as e:
#                 print(f"[PDF] generation error: {e}")
#                 _evolution_send_text(remote_jid, "Sorry, I couldn't generate the PDF. Please try again.")
#         else:
#             _evolution_send_text(remote_jid, "I don't have any sources from a previous question yet. Ask me something first, then request the sources.")

#     return JSONResponse(content={"ok": True, "handled": True})


# @app.get("/health")
# async def health():
#     return {"status": "ok"}


# # ---------------------------------------------------------------------------
# # Test UI: chat + simulate webhook
# # ---------------------------------------------------------------------------

# @app.post("/test/chat")
# async def test_chat(payload: dict = Body(...)):
#     """Test the agent without WhatsApp: same logic as webhook."""
#     user_id = (payload.get("user_id") or "test-user").strip() or "test-user"
#     message = (payload.get("message") or "").strip()
#     if not message:
#         return JSONResponse(content={"error": "message is required"}, status_code=400)
#     try:
#         response_text = get_agent_response(user_id, message)
#         return {"response": response_text, "user_id": user_id}
#     except Exception as e:
#         return JSONResponse(
#             content={"error": str(e), "response": "Something went wrong."},
#             status_code=500,
#         )


# @app.get("/test/sources-pdf")
# async def test_sources_pdf(user_id: str = "test-user"):
#     """Generate a PDF from the last retrieved sources for this user. Download for testing."""
#     import base64
#     from fastapi.responses import Response

#     docs = get_last_docs(user_id.strip() or "test-user")
#     if not docs:
#         return JSONResponse(
#             content={
#                 "error": "No sources yet. Send a question in Test chat first (e.g. \"How do I change my ringtone on iPhone?\"), then try again.",
#             },
#             status_code=400,
#         )
#     try:
#         b64 = create_sources_pdf(docs)
#         pdf_bytes = base64.b64decode(b64)
#         return Response(
#             content=pdf_bytes,
#             media_type="application/pdf",
#             headers={"Content-Disposition": "attachment; filename=Care_Support_Sources.pdf"},
#         )
#     except Exception as e:
#         return JSONResponse(content={"error": str(e)}, status_code=500)


# _DEMO_HTML = """
# <!DOCTYPE html>
# <html lang="en">
# <head>
# <meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
# <title>Saksham Care – WhatsApp RAG Bot</title>
# <style>
# *{box-sizing:border-box;margin:0;padding:0}
# body{font-family:'Segoe UI',system-ui,-apple-system,sans-serif;background:#0b141a;color:#e9edef;height:100vh;display:flex;flex-direction:column}
# /* header */
# .header{background:#202c33;padding:10px 16px;display:flex;align-items:center;gap:12px;border-bottom:1px solid #2a3942;flex-shrink:0}
# .avatar{width:40px;height:40px;border-radius:50%;background:#00a884;display:flex;align-items:center;justify-content:center;font-size:1.3rem;font-weight:700;color:#fff}
# .hdr-info{flex:1}
# .hdr-name{font-size:.95rem;font-weight:600;color:#e9edef}
# .hdr-status{font-size:.75rem;color:#8696a0;display:flex;align-items:center;gap:4px}
# .dot{width:7px;height:7px;border-radius:50%;display:inline-block}
# .dot.on{background:#00a884}.dot.off{background:#ea4335}
# .badge{font-size:.65rem;background:#00a884;color:#fff;border-radius:10px;padding:2px 8px;margin-left:8px;font-weight:600}
# .badge.wa{background:#25d366}
# /* chat area */
# .chat{flex:1;overflow-y:auto;padding:16px 60px;display:flex;flex-direction:column;gap:4px;background:url("data:image/svg+xml,%3Csvg width='60' height='60' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M0 0h60v60H0z' fill='%23091920'/%3E%3Cpath d='M30 10c1 0 2 .5 2 1.5s-1 1.5-2 1.5-2-.5-2-1.5.9-1.5 2-1.5z' fill='%230d2229' fill-opacity='.3'/%3E%3C/svg%3E")}
# .msg{max-width:75%;padding:6px 8px 18px;border-radius:8px;font-size:.9rem;line-height:1.45;position:relative;word-wrap:break-word;white-space:pre-wrap}
# .msg .time{position:absolute;right:6px;bottom:2px;font-size:.65rem;color:#8696a0}
# .msg.user{align-self:flex-end;background:#005c4b;border-top-right-radius:0}
# .msg.bot{align-self:flex-start;background:#202c33;border-top-left-radius:0}
# .msg.system{align-self:center;background:#182229;border-radius:6px;font-size:.78rem;color:#8696a0;padding:4px 12px;max-width:90%}
# .typing{align-self:flex-start;background:#202c33;border-radius:8px;border-top-left-radius:0;padding:10px 14px;display:none;gap:4px;align-items:center}
# .typing span{width:7px;height:7px;border-radius:50%;background:#8696a0;animation:blink 1.4s infinite both}
# .typing span:nth-child(2){animation-delay:.2s}.typing span:nth-child(3){animation-delay:.4s}
# @keyframes blink{0%,80%,100%{opacity:.3}40%{opacity:1}}
# /* input */
# .input-bar{background:#202c33;padding:8px 10px;display:flex;gap:8px;align-items:center;border-top:1px solid #2a3942;flex-shrink:0}
# .input-bar input{flex:1;background:#2a3942;border:none;border-radius:8px;padding:10px 14px;color:#e9edef;font-size:.9rem;outline:none}
# .input-bar input::placeholder{color:#8696a0}
# .send-btn{width:42px;height:42px;border-radius:50%;border:none;background:#00a884;cursor:pointer;display:flex;align-items:center;justify-content:center;flex-shrink:0}
# .send-btn:hover{background:#02b892}
# .send-btn:disabled{opacity:.5;cursor:not-allowed}
# .send-btn svg{fill:#fff;width:20px;height:20px}
# /* pdf bar */
# .pdf-bar{background:#182229;padding:6px 16px;display:flex;align-items:center;gap:10px;border-top:1px solid #2a3942;flex-shrink:0}
# .pdf-bar span{font-size:.78rem;color:#8696a0;flex:1}
# .pdf-btn{background:none;border:1px solid #00a884;color:#00a884;border-radius:6px;padding:5px 14px;font-size:.78rem;cursor:pointer;font-weight:600}
# .pdf-btn:hover{background:#00a884;color:#fff}
# .pdf-btn:disabled{opacity:.5;cursor:not-allowed}
# /* info panel */
# .info{background:#111b21;padding:6px 16px;border-top:1px solid #2a3942;flex-shrink:0;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:4px}
# .info a{color:#00a884;font-size:.72rem;text-decoration:none}.info a:hover{text-decoration:underline}
# .info .tag{font-size:.68rem;color:#8696a0}
# </style>
# </head>
# <body>
# <div class="header">
#   <div class="avatar">S</div>
#   <div class="hdr-info">
#     <div class="hdr-name">Saksham Care Bot</div>
#     <div class="hdr-status"><span class="dot on" id="dot"></span> <span id="statusTxt">Online</span><span class="badge wa">WhatsApp RAG</span></div>
#   </div>
# </div>

# <div class="chat" id="chat"></div>

# <div class="typing" id="typing"><span></span><span></span><span></span></div>

# <div class="pdf-bar">
#   <span>Last sources retrieved by the bot</span>
#   <button class="pdf-btn" id="pdfBtn" disabled>Download Sources PDF</button>
# </div>

# <div class="input-bar">
#   <input type="text" id="msgInput" placeholder="Type a message..." autocomplete="off" />
#   <button class="send-btn" id="sendBtn"><svg viewBox="0 0 24 24"><path d="M1.1 21.757l22.4-9.71L1.1 2.33l.003 7.567 16.07 2.143L1.103 14.19z"/></svg></button>
# </div>

# <div class="info">
#   <div class="tag">Saksham MVP &middot; LangChain RAG &middot; FAISS + BM25 Hybrid &middot; CRAG &middot; Evolution API</div>
#   <div><a href="/health">/health</a> &middot; <a href="/docs">/docs</a></div>
# </div>

# <script>
# const API = (location.origin && location.origin.startsWith('http')) ? location.origin : 'http://127.0.0.1:8000';
# const chat = document.getElementById('chat');
# const input = document.getElementById('msgInput');
# const sendBtn = document.getElementById('sendBtn');
# const typing = document.getElementById('typing');
# const pdfBtn = document.getElementById('pdfBtn');
# const dot = document.getElementById('dot');
# const statusTxt = document.getElementById('statusTxt');
# const userId = 'demo-' + Math.random().toString(36).slice(2,8);
# let hasSources = false;

# function now() { const d = new Date(); return d.getHours().toString().padStart(2,'0') + ':' + d.getMinutes().toString().padStart(2,'0'); }

# function addMsg(text, cls) {
#   const d = document.createElement('div');
#   d.className = 'msg ' + cls;
#   d.innerHTML = text.replace(/\\n/g,'<br>').replace(/\*\*(.*?)\*\*/g,'<b>$1</b>') + '<span class="time">' + now() + '</span>';
#   chat.appendChild(d);
#   chat.scrollTop = chat.scrollHeight;
# }

# function addSystem(text) {
#   const d = document.createElement('div');
#   d.className = 'msg system';
#   d.textContent = text;
#   chat.appendChild(d);
#   chat.scrollTop = chat.scrollHeight;
# }

# addSystem("Saksham Care Bot is online. This is the same RAG pipeline that runs on WhatsApp via Evolution API.");
# addMsg("Hi there! Welcome to Saksham Support. I'm here to help with your iPhone or Pixel questions, or to listen if something feels off or scam-related. How can I help you today?", "bot");

# async function send() {
#   const msg = input.value.trim();
#   if (!msg) return;
#   input.value = '';
#   addMsg(msg, 'user');
#   sendBtn.disabled = true;
#   typing.style.display = 'flex';
#   dot.className = 'dot on';
#   statusTxt.textContent = 'typing...';
#   chat.scrollTop = chat.scrollHeight;
#   try {
#     const r = await fetch(API + '/test/chat', {
#       method: 'POST',
#       headers: {'Content-Type':'application/json'},
#       body: JSON.stringify({user_id: userId, message: msg})
#     });
#     const data = await r.json();
#     typing.style.display = 'none';
#     if (r.ok && data.response) {
#       addMsg(data.response, 'bot');
#       hasSources = true;
#       pdfBtn.disabled = false;
#     } else {
#       addMsg(data.error || 'Something went wrong.', 'bot');
#     }
#   } catch(e) {
#     typing.style.display = 'none';
#     addMsg('Connection error. Is the server running?', 'bot');
#   }
#   sendBtn.disabled = false;
#   dot.className = 'dot on';
#   statusTxt.textContent = 'Online';
#   input.focus();
# }

# sendBtn.addEventListener('click', send);
# input.addEventListener('keydown', e => { if (e.key === 'Enter') send(); });

# pdfBtn.addEventListener('click', async () => {
#   if (!hasSources) return;
#   pdfBtn.disabled = true;
#   try {
#     const r = await fetch(API + '/test/sources-pdf?user_id=' + encodeURIComponent(userId));
#     if (r.ok) {
#       const blob = await r.blob();
#       const url = URL.createObjectURL(blob);
#       const a = document.createElement('a');
#       a.href = url; a.download = 'Care_Support_Sources.pdf'; a.click();
#       URL.revokeObjectURL(url);
#       addSystem("Sources PDF downloaded.");
#     } else {
#       addSystem("No sources yet. Ask a question first.");
#     }
#   } catch(e) { addSystem("PDF download failed."); }
#   pdfBtn.disabled = false;
# });

# input.focus();
# </script>
# </body>
# </html>
# """


# @app.get("/", response_class=HTMLResponse)
# async def demo_ui():
#     """Saksham Care WhatsApp RAG Bot — demo chat UI."""
#     return HTMLResponse(_DEMO_HTML)
