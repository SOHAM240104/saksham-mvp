<!-- # WhatsApp RAG Chatbot – POC

Minimal steps to run and test the Care RAG app as a WhatsApp chatbot.

## 1. Run the backend (one command)

```bash
cd "/Users/soham/Desktop/ multimodal"
chmod +x run_poc.sh
./run_poc.sh
```

Or manually:

```bash
cd "/Users/soham/Desktop/ multimodal"
export KMP_DUPLICATE_LIB_OK=TRUE   # fixes OpenMP crash on macOS
source venv/bin/activate
uvicorn server:app --host 0.0.0.0 --port 8000
```

## 2. Test in the browser (no WhatsApp needed)

1. Open **http://localhost:8000/** in your browser.
2. In **"Test chat"**: type a message (e.g. *How do I change my ringtone on iPhone?*) and click **Send**.
3. You should see the RAG agent’s reply. That’s the same logic used when someone messages on WhatsApp.

If this works, the app is ready to connect to WhatsApp.

## 3. Connect to WhatsApp (optional)

1. **Evolution API** must be running (e.g. at http://localhost:8080 – see DEPLOY.md).
2. In this project’s **.env** set:
   - `EVOLUTION_API_URL=http://localhost:8080`
   - `EVOLUTION_API_KEY=your-api-key`
   - `EVOLUTION_INSTANCE_NAME=saksham-care`
3. **Number for the bot**: Use a **virtual/dedicated number** for testing so your personal WhatsApp stays separate. Options: a second SIM, a VoIP number that supports WhatsApp verification (e.g. Twilio, MessageBird), or a cheap prepaid SIM. Link that number in Evolution (scan QR). Your personal number is only used to chat *with* the bot.
4. Create an instance and set the webhook to your backend (see DEPLOY.md). Expose your local backend so Evolution can reach it:
   - **Cloudflare Tunnel** (no account needed for quick tunnels): see [Cloudflare Tunnel](#cloudflare-tunnel) below.
   - **ngrok**: `ngrok http 8000` (requires account + authtoken).

## Cloudflare Tunnel

Use this when Evolution API needs to call your webhook from the internet (e.g. Evolution is cloud-hosted or on another network). No Cloudflare account needed for a quick tunnel.

1. **Install cloudflared** (one-time):
   ```bash
   brew install cloudflared
   ```
   Or download from [developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation](https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation).

2. **Start the tunnel** (with Care backend already running on port 8000):
   ```bash
   cloudflared tunnel --url http://localhost:8000
   ```
   It will print a URL like `https://xxxx-xx-xx-xx-xx.trycloudflare.com`.

3. **Set Evolution webhook** to that URL + `/webhook/evolution`:
   ```bash
   curl -X POST "http://localhost:8080/webhook/set/saksham-care" \
     -H "apikey: YOUR_EVOLUTION_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"webhook":{"enabled":true,"url":"https://YOUR-TUNNEL-URL.trycloudflare.com/webhook/evolution","events":["MESSAGES_UPSERT"],"byEvents":false,"base64":false}}'
   ```
   Replace `YOUR-TUNNEL-URL` and `YOUR_EVOLUTION_API_KEY` with the values from step 2 and your `.env`.

4. Keep **Care** (port 8000), **Evolution** (port 8080), and the **cloudflared** terminal running. When you message the linked WhatsApp number, Evolution will call your webhook via the tunnel and Care will reply.

**Note:** The quick tunnel URL changes each time you restart `cloudflared`. If you use a named tunnel or a custom domain with Cloudflare, you can reuse the same URL.

## Troubleshooting

- **"Failed to fetch"** – Backend not running. Start it with `./run_poc.sh`.
- **OMP / libomp abort** – Already handled by `KMP_DUPLICATE_LIB_OK=TRUE` in code and in `run_poc.sh`. If it still crashes, run: `export KMP_DUPLICATE_LIB_OK=TRUE` in the same terminal before `uvicorn`. -->
