# Deploy Care Assistant to Streamlit Community Cloud

## Prerequisites

1. **GitHub account** – [github.com](https://github.com)
2. **Streamlit Community Cloud account** – [share.streamlit.io](https://share.streamlit.io) (sign in with GitHub)
3. **OpenAI API key** – for the RAG chat
4. **Tesseract OCR binary** — for image-based scam detection

   - macOS (brew): `brew install tesseract`
   - Ubuntu: `sudo apt-get update && sudo apt-get install -y tesseract-ocr`

5. **WeasyPrint (PDF sources)** — the Python package is in `requirements.txt`, but WeasyPrint needs **system** libraries (Pango, Cairo, GObject) and **at least one installed font** (Fontconfig alone is not enough). Without TTFs, PDFs can render as blank white pages. The repo’s **`packages.txt`** includes **`fonts-dejavu-core`** for that reason. On **Streamlit Community Cloud**, commit **`packages.txt`** in the repo root (one **package name per line**, **no `#` comment lines** — apt treats every line as a package). If native libs are missing, the app will show an error when it tries to build the sources PDF. Locally on macOS, Homebrew or distro packages supply the libs and fonts.

## 1. Push your app to GitHub

From your project folder:

```bash
git init
git add app.py system_prompt.txt requirements.txt .streamlit/ chroma_db_apple/ chroma_db_google/
git commit -m "Care Assistant app for Streamlit Cloud"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

**Important:** Include the `chroma_db_apple/` and `chroma_db_google/` folders so the deployed app has the support data. If they are large, run `ingest.py` once locally to build them, then add and commit them.

## 2. Deploy on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io).
2. Click **“New app”**.
3. Choose your GitHub repo, branch (e.g. `main`), and set **Main file path** to `app.py`.
4. Click **“Advanced settings”** and add your secret:

   ```toml
   OPENAI_API_KEY = "sk-your-openai-api-key-here"
   ```

5. Click **“Deploy!”**.

The app will build and start. The first run can take a couple of minutes while dependencies install.

## 3. After deployment

- **Secrets:** Change or add secrets in the app’s **Settings** (three dots) → **Secrets**.
- **Updates:** Push to the connected branch; Streamlit Cloud will redeploy.
- **Logs:** Use **Manage app** → **Logs** to debug errors.

## Local development

```bash
python -m venv venv
source venv/bin/activate   # or: venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Create a `.env` file with `OPENAI_API_KEY=sk-...`. Run the app:

```bash
streamlit run app.py
```

To refresh the Chroma DBs (e.g. after changing URLs in `ingest.py`):

```bash
python ingest.py
```

Then commit the updated `chroma_db_apple/` and `chroma_db_google/` if you want the same data on Streamlit Cloud.

## WhatsApp (Evolution API) backend

### From scratch (step-by-step, no Docker required)

Follow these steps in order. You have two ways to get Evolution API: **hosted (no Docker)** or **self-hosted (with Docker)**.

---

**Step 0 – Prerequisites on your machine**

- Python 3.9+ and pip
- This project with dependencies installed:  
  `pip install -r requirements.txt`
- A `.env` file with at least:  
  `OPENAI_API_KEY=sk-your-key`
- Vector DB ready (run `python ingest.py` once if you haven’t)

---

**Step 1 – Get Evolution API (choose one)**

**Path A – Hosted (no Docker)**  
1. Sign up with a hosted Evolution API provider (e.g. [Evolution API Cloud](https://docs.evoapicloud.com/) or search for “Evolution API hosted”).  
2. Create a project / instance in their dashboard.  
3. Note your **API base URL** and **API key**.  
4. You’re done with “running” Evolution — go to Step 2.

**Path B – Self-hosted (with Docker)**  
1. Install Docker: [Install Docker](https://docs.docker.com/get-docker/).  
2. Clone and run Evolution API (they need PostgreSQL and Redis; use their official [Docker install](https://doc.evolution-api.com/v2/en/install/docker)):

   ```bash
   # Example: in a new folder
   git clone https://github.com/EvolutionAPI/evolution-api.git
   cd evolution-api
   # Create .env with AUTHENTICATION_API_KEY=your-secret and DB/Redis vars (see their .env.example)
   docker compose up -d
   ```

3. Your Evolution API URL is `http://localhost:8080` (or the host/port you expose).  
4. Go to Step 2.

---

**Step 2 – Run your Care backend**

1. Open a terminal in this project folder:

   ```bash
   cd "/Users/soham/Desktop/ multimodal"
   source venv/bin/activate   # if you use a venv
   uvicorn server:app --host 0.0.0.0 --port 8000
   ```

2. Leave this running. Your webhook will be at:  
   `http://YOUR_IP:8000/webhook/evolution`  
   For Evolution API Cloud (or any server on the internet) to reach it, your Care backend must be **publicly reachable** (e.g. deploy on a VPS/cloud and use HTTPS, or use [ngrok](https://ngrok.com/) for testing: `ngrok http 8000` → use the HTTPS URL they give you).

---

**Step 3 – Add Evolution to `.env`**

In this project’s `.env`, add (use the URL and key from Step 1):

```env
EVOLUTION_API_URL=https://your-evolution-api-url.com
EVOLUTION_API_KEY=your-evolution-api-key
EVOLUTION_INSTANCE_NAME=saksham-care
```

Use the same **instance name** in the next step.

---

**Step 4 – Create a WhatsApp instance (terminal)**

Replace `YOUR_EVOLUTION_API_URL`, `YOUR_GLOBAL_API_KEY`, and the instance name if you changed it:

```bash
curl -X POST "YOUR_EVOLUTION_API_URL/instance/create" \
  -H "apikey: YOUR_GLOBAL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"instanceName": "saksham-care", "integration": "WHATSAPP-BAILEYS", "qrcode": true}'
```

You should get a QR code in the response (or a link to the instance dashboard to show the QR).

---

**Step 5 – Connect WhatsApp**

- Open WhatsApp on your phone → Settings → Linked devices → Link a device.  
- Scan the QR code from Step 4 (or from the Evolution API dashboard).  
- Wait until it says “Connected”.

---

**Step 6 – Set the webhook (terminal)**

Replace:

- `YOUR_EVOLUTION_API_URL` and `YOUR_GLOBAL_API_KEY` as before  
- `https://your-public-care-backend.com` with the **public URL** of your Care backend (from Step 2), including `/webhook/evolution`

```bash
curl -X POST "YOUR_EVOLUTION_API_URL/webhook/set/saksham-care" \
  -H "apikey: YOUR_GLOBAL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "enabled": true,
    "url": "https://your-public-care-backend.com/webhook/evolution",
    "webhookByEvents": false,
    "webhookBase64": false,
    "events": ["MESSAGES_UPSERT"]
  }'
```

---

**Step 7 – Test**

1. Send a WhatsApp message to the number that’s linked to your Evolution instance.  
2. Your Care backend should receive it and reply.  
3. If it doesn’t: check that the Care backend is running (Step 2), the webhook URL is reachable from the internet, and `.env` has the correct Evolution URL, key, and instance name.

---

### Do I need to run something before using Evolution API?

**Yes.** Evolution API is a **separate service** that talks to WhatsApp. It must be running *before* your Care backend can receive or send WhatsApp messages.

You have two options:

| Option | What you do |
|--------|-------------|
| **A. Hosted Evolution API** | Sign up with a provider that runs Evolution API for you. They give you a URL and API key. You don’t run anything for Evolution — only your Care backend. |
| **B. Self‑host Evolution API** | Run Evolution API yourself (e.g. with Docker). You need it running first, then run your Care backend, then use the terminal (curl) to create an instance and set the webhook. |

Everything below (creating instance, setting webhook) is done **from the terminal** with `curl` once Evolution API is available (hosted or self‑hosted).

---

To run the FastAPI webhook server for WhatsApp:

1. In `.env`, set:
   - `EVOLUTION_API_URL` – base URL of your Evolution API (e.g. `https://your-evolution-api.com`)
   - `EVOLUTION_API_KEY` – Global API Key for Evolution API
   - `EVOLUTION_INSTANCE_NAME` – instance name configured in Evolution API

2. Start the server:
   ```bash
   uvicorn server:app --host 0.0.0.0 --port 8000
   ```

3. Configure Evolution API (see below) so its webhook points to `https://your-server/webhook/evolution`.

**Run order from the terminal:**

1. **Evolution API** – If self‑hosting: start it (e.g. `docker compose up -d` in Evolution’s project; see [Evolution API Docker docs](https://doc.evolution-api.com/v2/en/install/docker)). If using a hosted provider, you only need the URL and API key.
2. **Your Care backend** – From this project folder:  
   `uvicorn server:app --host 0.0.0.0 --port 8000`
3. **Create instance + set webhook** – Use the `curl` commands in the next section (same terminal or another).

---

### How to configure Evolution API

You need a running Evolution API server (self‑hosted or a hosted provider). Then:

#### 1. Create an instance

Create a WhatsApp instance with the name you will use in `EVOLUTION_INSTANCE_NAME` (e.g. `saksham-care`):

```bash
curl -X POST "https://YOUR_EVOLUTION_API_URL/instance/create" \
  -H "apikey: YOUR_GLOBAL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "instanceName": "saksham-care",
    "integration": "WHATSAPP-BAILEYS",
    "qrcode": true
  }'
```

- **instanceName** – must match `EVOLUTION_INSTANCE_NAME` in your `.env`.
- **integration** – `WHATSAPP-BAILEYS` (multi-device) or `WHATSAPP-BUSINESS`.
- **qrcode: true** – so you get a QR code to link WhatsApp.

Use the returned QR code (or the instance’s “Connect” flow in the Evolution API UI) to link your WhatsApp number.

#### 2. Set the webhook for incoming messages

Tell Evolution to send new messages to your Care backend. Replace `YOUR_EVOLUTION_API_URL`, `YOUR_GLOBAL_API_KEY`, `saksham-care`, and `https://your-backend.com` with your values:

```bash
curl -X POST "https://YOUR_EVOLUTION_API_URL/webhook/set/saksham-care" \
  -H "apikey: YOUR_GLOBAL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "enabled": true,
    "url": "https://your-backend.com/webhook/evolution",
    "webhookByEvents": false,
    "webhookBase64": false,
    "events": ["MESSAGES_UPSERT"]
  }'
```

- **url** – your FastAPI server’s public URL + `/webhook/evolution` (must be HTTPS and reachable from the internet).
- **events** – `MESSAGES_UPSERT` is required so your server receives every new message (and can reply via the API).

#### 3. Confirm

- **Check webhook:**  
  `GET https://YOUR_EVOLUTION_API_URL/webhook/find/saksham-care`  
  with header `apikey: YOUR_GLOBAL_API_KEY` — should show your URL and `MESSAGES_UPSERT`.
- **Test:** Send a WhatsApp message to the linked number; your backend should get a POST at `/webhook/evolution` and reply with the Care assistant.

#### 4. Env summary

| Variable | Example | Description |
|----------|---------|-------------|
| `EVOLUTION_API_URL` | `https://evolution.example.com` | Evolution API base URL (no trailing slash). |
| `EVOLUTION_API_KEY` | Your global API key | Same key used in the `apikey` header above. |
| `EVOLUTION_INSTANCE_NAME` | `saksham-care` | The `instanceName` you created. |
