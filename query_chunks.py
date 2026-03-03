#!/usr/bin/env python3
"""
Run a few queries against the vector DB and print retrieved chunks in the terminal.
Usage: python query_chunks.py   (uses .env for OPENAI_API_KEY)
"""
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

VECTOR_DB_PATH = "./care_vector_db"
PLATFORM_APPLE = "IOS18"
PLATFORM_GOOGLE = "PIXEL"

# Example queries to run
QUERIES = [
    ("how do I restart my iPhone", PLATFORM_APPLE),
    ("restart my Pixel phone", PLATFORM_GOOGLE),
    ("turn on camera", PLATFORM_APPLE),
    ("block a number", PLATFORM_GOOGLE),
]

def main():
    if not os.path.exists(VECTOR_DB_PATH):
        print("Run ingest first. DB not found:", VECTOR_DB_PATH)
        return

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embeddings,
    )

    for query, platform in QUERIES:
        print("\n" + "=" * 70)
        print(f"QUERY: \"{query}\"  |  FILTER: platform={platform}")
        print("=" * 70)

        search_kwargs = {"k": 4}
        search_kwargs["filter"] = {"platform": platform}
        retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
        docs = retriever.invoke(query)

        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "?")
            title = doc.metadata.get("title", "")[:60]
            text_preview = (doc.page_content[:400] + "..." if len(doc.page_content) > 400 else doc.page_content)
            print(f"\n--- Chunk {i} ---")
            print(f"  source: {source}")
            print(f"  title:  {title}")
            print(f"  text:\n  {text_preview.replace(chr(10), chr(10) + '  ')}")
        print()

if __name__ == "__main__":
    main()
