# #!/usr/bin/env bash
# # COMMENTED OUT (Streamlit-only). WhatsApp RAG POC – run backend + test UI.
# set -e
# cd "$(dirname "$0")"

# # Avoid OpenMP crash on macOS when using FAISS/sentence-transformers
# export KMP_DUPLICATE_LIB_OK=TRUE

# if [ ! -d "venv" ]; then
#   echo "Creating venv..."
#   python3 -m venv venv
# fi
# source venv/bin/activate

# if ! python -c "import fastapi" 2>/dev/null; then
#   echo "Installing dependencies..."
#   pip install -r requirements.txt -q
# fi

# echo ""
# echo "  Care backend (WhatsApp RAG POC)"
# echo "  -------------------------------"
# echo "  Test UI:  http://localhost:8000/"
# echo "  Health:   http://localhost:8000/health"
# echo "  API docs: http://localhost:8000/docs"
# echo ""
# echo "  Use the Test UI to send messages and verify the RAG agent."
# echo "  To use with real WhatsApp: set Evolution API in .env and configure webhook."
# echo ""

# uvicorn server:app --host 0.0.0.0 --port 8000
