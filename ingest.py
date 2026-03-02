import os
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import FireCrawlLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# -------------------------
# DATABASE PATHS
# -------------------------
APPLE_DB = "./chroma_db_apple"
GOOGLE_DB = "./chroma_db_google"

# -------------------------
# CURATED APPLE SUPPORT URLS
# -------------------------
APPLE_URLS = [

    # iOS Update
    "https://support.apple.com/en-in/guide/iphone/iphfed2c4091/18.0/ios/18.0",

    # Control Center
    "https://support.apple.com/en-in/guide/iphone/iph59095ec58/18.0/ios/18.0",

    # Battery & Charging
    "https://support.apple.com/en-in/guide/iphone/iph3e2e2cdc/18.0/ios/18.0",

    # Face ID
    "https://support.apple.com/en-in/guide/iphone/iph14c9c9d2/18.0/ios/18.0",

    # Wi-Fi
    "https://support.apple.com/en-in/guide/iphone/iph3dd5f213/18.0/ios/18.0",

    # Accessibility
    "https://support.apple.com/en-in/guide/iphone/iph9c38e6a2c/18.0/ios/18.0",

    # Reset Settings
    "https://support.apple.com/en-in/guide/iphone/iph6f3c3c6f5/18.0/ios/18.0",

    # Emergency SOS
    "https://support.apple.com/en-in/guide/iphone/iph08022b192/18.0/ios/18.0"
]

# -------------------------
# CURATED PIXEL SUPPORT URLS
# -------------------------
GOOGLE_URLS = [

    # Troubleshooting
    "https://support.google.com/pixelphone/answer/14140287?hl=en",

    # Android Update
    "https://support.google.com/pixelphone/answer/7680439?hl=en",

    # Battery Saver
    "https://support.google.com/pixelphone/answer/6187458?hl=en",

    # Wi-Fi Issues
    "https://support.google.com/pixelphone/answer/6183600?hl=en",

    # Factory Reset
    "https://support.google.com/pixelphone/answer/4596836?hl=en",

    # Accessibility
    "https://support.google.com/pixelphone/answer/6006564?hl=en",

    # Calls & Contacts
    "https://support.google.com/pixelphone/answer/2811745?hl=en"
]

# -------------------------
# EMBEDDING MODEL
# -------------------------
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


# -------------------------
# INGEST FUNCTION
# -------------------------
def ingest_urls(urls, db_path, device_type):

    print(f"\n🚀 Ingesting {device_type} documentation...")

    # Clear existing DB
    if os.path.exists(db_path):
        print("🧹 Clearing existing database...")
        shutil.rmtree(db_path)

    all_docs = []
    indexed_urls = []

    for url in urls:

        print(f"🔎 Scraping: {url}")

        loader = FireCrawlLoader(
            api_key=os.environ.get("FIRECRAWL_API_KEY"),
            url=url,
            mode="scrape"  # 🔥 using scrape for direct content pages
        )

        docs = loader.load()

        for doc in docs:
            doc.metadata["device_type"] = device_type
            doc.metadata["source"] = url  # force correct source metadata
            all_docs.append(doc)

        indexed_urls.append(url)

    print(f"📄 Total pages indexed: {len(indexed_urls)}")

    # Save crawl report
    report_file = f"{device_type.lower()}_indexed_pages.txt"
    with open(report_file, "w") as f:
        for url in indexed_urls:
            f.write(url + "\n")

    print(f"📝 Indexed page list saved to {report_file}")

    # -------------------------
    # CHUNKING
    # -------------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(all_docs)

    print(f"✂️ Total chunks created: {len(chunks)}")

    # -------------------------
    # STORE IN CHROMA
    # -------------------------
    Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=db_path
    )

    print(f"✅ {device_type} database created at {db_path}")


# -------------------------
# MAIN
# -------------------------
def main():
    ingest_urls(APPLE_URLS, APPLE_DB, "APPLE")
    ingest_urls(GOOGLE_URLS, GOOGLE_DB, "GOOGLE")
    print("\n🎉 All databases built successfully.")


if __name__ == "__main__":
    main()