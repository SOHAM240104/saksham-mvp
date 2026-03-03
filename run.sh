#!/usr/bin/env bash
# Build support DB if missing, then start the app (for local or Streamlit Cloud).
set -e
if [ ! -d "./care_vector_db" ]; then
  echo "Building support database (this may take a few minutes)..."
  python ingest.py
fi
exec streamlit run app.py --server.headless true "$@"
