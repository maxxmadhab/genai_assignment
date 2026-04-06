"""
Legal RAG - Offline Indexer
============================
Run this ONCE to index all your legal PDFs.
Place all PDFs in the ./legal_docs/ folder first.

Usage:
    python index_documents.py

This will create:
    ./vector_store/faiss/       ← FAISS index
    ./vector_store/chroma/      ← Chroma index
"""

import os
import glob
from pathlib import Path

# ──────────────────────────────────────────────
# CONFIG — edit these as needed
# ──────────────────────────────────────────────
PDF_FOLDER        = "./legal_docs"          # Put all your PDFs here
FAISS_STORE_PATH  = "./vector_store/faiss"
CHROMA_STORE_PATH = "./vector_store/chroma"

# Chunking
SMALL_CHUNK_SIZE    = 500
SMALL_CHUNK_OVERLAP = 50
LARGE_CHUNK_SIZE    = 1000
LARGE_CHUNK_OVERLAP = 100

# Embedding models
EMB_SMALL = "all-MiniLM-L6-v2"     # used for FAISS
EMB_LARGE = "all-mpnet-base-v2"     # used for Chroma

# ──────────────────────────────────────────────
# STEP 1: Load all PDFs
# ──────────────────────────────────────────────
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

pdf_files = glob.glob(os.path.join(PDF_FOLDER, "**/*.pdf"), recursive=True) + \
            glob.glob(os.path.join(PDF_FOLDER, "*.pdf"))
pdf_files = list(set(pdf_files))  # deduplicate

if not pdf_files:
    print(f"\n❌ No PDFs found in '{PDF_FOLDER}/'")
    print("   Create the folder and place your legal PDFs inside it, then re-run.\n")
    exit(1)

print(f"\n📂 Found {len(pdf_files)} PDF(s):")
docs = []
for pdf in pdf_files:
    print(f"   Loading: {pdf}")
    loader = PyPDFLoader(pdf)
    loaded = loader.load()
    # tag each chunk with its source filename
    for d in loaded:
        d.metadata["source"] = os.path.basename(pdf)
    docs.extend(loaded)

print(f"\n✅ Total pages loaded: {len(docs)}")

# ──────────────────────────────────────────────
# STEP 2: Chunk documents
# ──────────────────────────────────────────────
splitter_small = RecursiveCharacterTextSplitter(
    chunk_size=SMALL_CHUNK_SIZE,
    chunk_overlap=SMALL_CHUNK_OVERLAP,
)
splitter_large = RecursiveCharacterTextSplitter(
    chunk_size=LARGE_CHUNK_SIZE,
    chunk_overlap=LARGE_CHUNK_OVERLAP,
)

docs_small = splitter_small.split_documents(docs)
docs_large = splitter_large.split_documents(docs)

print(f"✅ Chunks for FAISS  (small): {len(docs_small)}")
print(f"✅ Chunks for Chroma (large): {len(docs_large)}")

# ──────────────────────────────────────────────
# STEP 3: Build & Save FAISS index
# ──────────────────────────────────────────────
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, Chroma

print(f"\n🔨 Building FAISS index with '{EMB_SMALL}'...")
emb_small = HuggingFaceEmbeddings(model_name=EMB_SMALL)
db_faiss = FAISS.from_documents(docs_small, emb_small)

os.makedirs(FAISS_STORE_PATH, exist_ok=True)
db_faiss.save_local(FAISS_STORE_PATH)
print(f"✅ FAISS index saved → {FAISS_STORE_PATH}")

# ──────────────────────────────────────────────
# STEP 4: Build & Save Chroma index
# ──────────────────────────────────────────────
print(f"\n🔨 Building Chroma index with '{EMB_LARGE}'...")
emb_large = HuggingFaceEmbeddings(model_name=EMB_LARGE)

os.makedirs(CHROMA_STORE_PATH, exist_ok=True)
db_chroma = Chroma.from_documents(
    docs_large,
    emb_large,
    persist_directory=CHROMA_STORE_PATH,
)
print(f"✅ Chroma index saved → {CHROMA_STORE_PATH}")

# ──────────────────────────────────────────────
# STEP 5: Summary
# ──────────────────────────────────────────────
print("\n" + "="*50)
print("  INDEXING COMPLETE")
print("="*50)
print(f"  PDFs indexed : {len(pdf_files)}")
print(f"  Total pages  : {len(docs)}")
print(f"  FAISS chunks : {len(docs_small)}")
print(f"  Chroma chunks: {len(docs_large)}")
print(f"  FAISS path   : {FAISS_STORE_PATH}")
print(f"  Chroma path  : {CHROMA_STORE_PATH}")
print("\n  You can now run:  streamlit run legal_rag_app.py\n")