import os
import shutil
import argparse
from dotenv import load_dotenv
from langchain_community.document_loaders import FireCrawlLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# -------------------------
# DATABASE PATH
# -------------------------
MAIN_DB = "./care_vector_db"

# -------------------------
# APPLE URLS (YOUR EXACT LIST)
# -------------------------
APPLE_URLS = [
    "https://support.apple.com/en-in/guide/iphone/iph4fd8a0b89/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph2968440de/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iphcfadf0701/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph62faab6a4/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph14c31d991/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph9bfec93b1/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iphca3d8b4e3/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph6d50ec543/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph8739025dd/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph3c9951d7/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph16ecebf48/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph6cfaf98b6/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph0e5ca7dd3/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph1ac0b35f/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph14a867ae/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph315e0d58d/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph6a6decb13/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph9e04f3be2/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph9ccdd1bab/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph76d37ce7d/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph1e466ebee/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph07c867f28/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph1ac0b4af/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph3c8f64e92/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph5fe280d90/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph584ea27f5/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph75f461ff0/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph35c335575/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph060d99f5e/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph0ede5b8d6/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph629d2cd37/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph70107aec2/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph22d98bbca/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iphbd435673d/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph37fdd714b/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph4ce326e9d/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph9a847efc7/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph7d116e557/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/ipha0706e2bc/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph1d019df4e/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph60d026b44/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph59095ec58/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph8f357526d/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph21a030ae3/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph3bf19d7b9/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iphc61044c11/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iphe9b48b89e/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iphf9219d8c9/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iphfaf30bdbd/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iphea8b95631/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph7f9e64962/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph2ab28320d/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph4cad323fe/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph5f869c0af/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph7df5983ee/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph7778f2888/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph4fff7afb5/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iphfc2cd4c5f/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iphf699c1ffc/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iphe9d46e90f/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph21addc265/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph120bd1b3b/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iphb9c2b948f/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iphf02627316/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iphfd5b616b5/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph9b7f53382/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph3e2e23d1/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph1fbef4daa/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph3afc3b3fc/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph212965adb/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph4df1c0dad/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph6ef6ef377/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph220ea8dca/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph206c570e3/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iphe7aa3336/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph4bda412a5/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph6a8e168cb/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph6013e96f4/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph0db74c881/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iphaff1d606/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iphc23ad3473/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iphb3100d149/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph392d77d5f/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iphcd5447866/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iphcd8d3c813/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iphf28f17237/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph29dbe3fb6/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph17dc31049/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph7fe7a50a7/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph2e42d3117/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iphe7f70fba4/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph9c11382ff/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph23f4d9aa9/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iphcce2770fd/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iphe22408524/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iphc57feab64/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iph02f94fc1c/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iphe0990f7bb/18.0/ios/18.0",
    "https://support.apple.com/en-in/guide/iphone/iphd5300a341/18.0/ios/18.0",
]

# -------------------------
# GOOGLE PIXEL URLS (YOUR EXACT LIST)
# -------------------------
GOOGLE_URLS = [
    # --- Troubleshooting & Basics ---
    "https://support.google.com/pixelphone/answer/14140287",  # Fix common issues
    "https://support.google.com/pixelphone/answer/12967594",  # Get around your phone
    "https://support.google.com/pixelphone/answer/7158570",   # How to use the Camera
    "https://support.google.com/pixelphone/answer/15199831",  # Find apps
    "https://support.google.com/pixelphone/answer/14116441",  # Navigation basics
    "https://support.google.com/pixelphone/answer/7535206",   # Update Android
    "https://support.google.com/pixelphone/answer/6111329",  # Connect to Wi-Fi
    "https://support.google.com/pixelphone/answer/7444033",   # Bluetooth help
    "https://support.google.com/pixelphone/answer/2819525",  # Contacts
    "https://support.google.com/pixelphone/answer/2818748",  # Phone calls
    "https://support.google.com/pixelphone/answer/7680439",   # System updates
    "https://support.google.com/pixelphone/answer/14782427",  # Personal Safety app
    "https://support.google.com/pixelphone/answer/6183600",   # Wi-Fi troubleshooting
    "https://support.google.com/pixelphone/answer/6187458",   # Battery Saver
    # --- Senior Specific (High Priority) ---
    "https://support.google.com/pixelphone/answer/6006564",   # Accessibility Overview
    "https://support.google.com/pixelphone/answer/6122841",   # Font & Display Size
    "https://support.google.com/pixelphone/answer/12913009",  # Using the Magnifier
    "https://support.google.com/pixelphone/answer/9316333",   # Medical ID & Emergency info
    "https://support.google.com/pixelphone/answer/7283669",   # Using the Google Assistant (Voice)
    "https://support.google.com/pixelphone/answer/2844832",   # Adjusting Volume & Vibration
    "https://support.google.com/pixelphone/answer/2781850",   # Changing Wallpapers
    "https://support.google.com/pixelphone/answer/15182154",  # Using Live Translate
    "https://support.google.com/pixelphone/answer/13202895",  # Adaptive Brightness
    # --- Safety & Anti-Scam ---
    "https://support.google.com/pixelphone/answer/7055029",   # Emergency SOS
    "https://support.google.com/pixelphone/answer/9118387",   # Screen calls (Spam protection)
    "https://support.google.com/pixelphone/answer/2819524",   # Blocking numbers
    "https://support.google.com/pixelphone/answer/4596836",   # Factory Reset
    "https://support.google.com/pixelphone/answer/9218411",   # Find your phone
    # --- Hardware & Maintenance ---
    "https://support.google.com/pixelphone/answer/6187455",   # Charging basics
    "https://support.google.com/pixelphone/answer/6090599",   # Cleaning your phone
    "https://support.google.com/pixelphone/answer/13675043",   # Identifying your Pixel model
    "https://support.google.com/pixelphone/answer/7106961",   # SIM card help
]

# -------------------------
# EMBEDDINGS
# -------------------------
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


def ingest_platform(urls, platform_name, clear_db=False):
    print(f"\n🚀 Ingesting {platform_name} documentation...")

    if clear_db and os.path.exists(MAIN_DB):
        print("🧹 Clearing existing database...")
        shutil.rmtree(MAIN_DB)

    all_docs = []
    success_urls = []

    for url in urls:
        print(f"🔎 Scraping: {url}")
        try:
            loader = FireCrawlLoader(
                api_key=os.environ.get("FIRECRAWL_API_KEY"),
                url=url,
                mode="scrape"
            )

            docs = loader.load()

            for doc in docs:
                doc.metadata["platform"] = platform_name
                doc.metadata["source"] = url
                all_docs.append(doc)

            success_urls.append(url)

        except Exception as e:
            print(f"❌ Failed to scrape {url}: {e}")

    print(f"📄 Successfully indexed {len(success_urls)} URLs")

    if not all_docs:
        print("⚠️ No documents loaded.")
        return

    # -------------------------
    # CHUNKING
    # -------------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(all_docs)
    print(f"✂️ Created {len(chunks)} chunks")

    # -------------------------
    # STORE IN CHROMA
    # -------------------------
    vectorstore = Chroma(
        persist_directory=MAIN_DB,
        embedding_function=embeddings
    )

    vectorstore.add_documents(chunks)
    # Chroma with persist_directory auto-persists; no .persist() in current API

    print(f"✅ {platform_name} data stored in {MAIN_DB}")

    # Save crawl report
    report_file = f"{platform_name.lower()}_indexed_pages.txt"
    with open(report_file, "w") as f:
        for url in success_urls:
            f.write(url + "\n")


def main():
    parser = argparse.ArgumentParser(description="Ingest Apple (iOS 18) and/or Google Pixel docs into vector DB.")
    parser.add_argument(
        "--pixel-only",
        action="store_true",
        help="Only ingest Pixel; do not clear DB or run Apple. Use when Apple is already done.",
    )
    args = parser.parse_args()

    if args.pixel_only:
        # Continue from where you left off: only add Pixel to existing DB
        if not os.path.exists(MAIN_DB):
            print("⚠️ No existing DB found. Run without --pixel-only first to ingest Apple.")
            return
        ingest_platform(GOOGLE_URLS, "PIXEL", clear_db=False)
    else:
        # Full run: clear, then Apple, then Pixel
        ingest_platform(APPLE_URLS, "IOS18", clear_db=True)
        ingest_platform(GOOGLE_URLS, "PIXEL")

    print("\n🎉 Ingest complete.")


if __name__ == "__main__":
    main()