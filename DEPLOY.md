# Deploy Care Assistant to Streamlit Community Cloud

## Prerequisites

1. **GitHub account** – [github.com](https://github.com)
2. **Streamlit Community Cloud account** – [share.streamlit.io](https://share.streamlit.io) (sign in with GitHub)
3. **OpenAI API key** – for the RAG chat

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
