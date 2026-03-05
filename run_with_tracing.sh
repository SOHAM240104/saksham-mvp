#!/usr/bin/env bash
# Run the Streamlit app with LangSmith tracing enabled.
# Set LANGSMITH_* and OPENAI_API_KEY in .env (see .env.example), or export them here.

set -e
cd "$(dirname "$0")"

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

# Optional: force tracing on if not in .env
export LANGSMITH_TRACING="${LANGSMITH_TRACING:-true}"
export LANGSMITH_PROJECT="${LANGSMITH_PROJECT:-pr-earnest-interval-32}"
export LANGSMITH_ENDPOINT="${LANGSMITH_ENDPOINT:-https://api.smith.langchain.com}"

echo "LangSmith tracing: ${LANGSMITH_TRACING} (project: ${LANGSMITH_PROJECT})"
exec streamlit run app.py "$@"
