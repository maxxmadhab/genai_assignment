"""
Legal RAG System - Streamlit App
=================================
Pre-requisite: Run index_documents.py first to build the vector stores.

Run with: streamlit run legal_rag_app.py
"""

import os
import streamlit as st

# ──────────────────────────────────────────────
# CONFIG — must match index_documents.py
# ──────────────────────────────────────────────
FAISS_STORE_PATH  = "./vector_store/faiss"
CHROMA_STORE_PATH = "./vector_store/chroma"
EMB_SMALL = "all-MiniLM-L6-v2"
EMB_LARGE = "all-mpnet-base-v2"

# ──────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Legal RAG Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Crimson+Pro:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;500&display=swap');
    html, body, [class*="css"] { font-family: 'Crimson Pro', Georgia, serif; }
    .stApp { background: #0f0e0c; color: #e8e0d0; }
    .main-title { font-size: 2.8rem; font-weight: 600; color: #c9a84c; letter-spacing: -0.5px; }
    .subtitle { font-size: 1.1rem; color: #8a7f6e; font-style: italic; margin-top: 0; }
    .section-header { font-size: 1.3rem; font-weight: 600; color: #c9a84c;
        border-bottom: 1px solid #2a2820; padding-bottom: 6px; margin-top: 1.5rem; }
    .answer-box { background: #1a1915; border: 1px solid #2e2c26;
        border-left: 3px solid #c9a84c; border-radius: 6px; padding: 1rem 1.2rem;
        font-size: 1.05rem; line-height: 1.7; color: #ddd5c0; margin-top: 0.5rem; }
    .source-chip { display: inline-block; background: #252319; border: 1px solid #3a3828;
        color: #9a9080; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem;
        padding: 2px 8px; border-radius: 3px; margin: 2px 3px; }
    .stButton > button { background: #c9a84c; color: #0f0e0c; border: none;
        font-family: 'Crimson Pro', serif; font-size: 1rem; font-weight: 600;
        padding: 0.5rem 1.5rem; border-radius: 4px; width: 100%; }
    .stButton > button:hover { background: #e0be6a; }
    .metric-card { background: #1a1915; border: 1px solid #2e2c26;
        border-radius: 6px; padding: 0.8rem 1rem; text-align: center; }
    .metric-value { font-size: 1.8rem; font-weight: 600; color: #c9a84c; }
    .metric-label { font-size: 0.85rem; color: #6a6050; font-style: italic; }
    .ready-badge { background: #1a2e1a; border: 1px solid #2a4a2a; color: #6abf6a;
        border-radius: 4px; padding: 0.5rem 1rem; font-size: 0.9rem; margin-bottom: 1rem; }
    .error-badge { background: #2e1a1a; border: 1px solid #4a2a2a; color: #bf6a6a;
        border-radius: 4px; padding: 0.5rem 1rem; font-size: 0.9rem; }
    .stTextInput > div > div > input, .stTextArea > div > div > textarea {
        background: #1a1915 !important; color: #e8e0d0 !important;
        border: 1px solid #3a3828 !important; font-family: 'Crimson Pro', serif !important;
        font-size: 1rem !important; }
    .stRadio > div { color: #e8e0d0; }
    [data-testid="stSidebar"] { background: #0c0b09; border-right: 1px solid #1e1d18; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Cached: Load pre-built vector stores
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_retrievers():
    """Load FAISS and Chroma from saved disk indexes."""
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_chroma import Chroma

    emb_small = HuggingFaceEmbeddings(model_name=EMB_SMALL)
    emb_large = HuggingFaceEmbeddings(model_name=EMB_LARGE)

    db_faiss = FAISS.load_local(
        FAISS_STORE_PATH,
        emb_small,
        allow_dangerous_deserialization=True,   # safe — your own local files
    )
    db_chroma = Chroma(
        persist_directory=CHROMA_STORE_PATH,
        embedding_function=emb_large,
    )

    retriever_faiss  = db_faiss.as_retriever(search_kwargs={"k": 3})
    retriever_chroma = db_chroma.as_retriever(search_kwargs={"k": 3})
    return retriever_faiss, retriever_chroma


@st.cache_resource(show_spinner=False)
def load_llm():
    """Load the text-generation pipeline (cached across sessions)."""
    from transformers import pipeline
    return pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_new_tokens=300,
        do_sample=False,
        temperature=1.0,
        repetition_penalty=1.1,
    )


def rag_pipeline(query: str, retriever, generator) -> tuple[str, list[str]]:
    """Retrieve relevant chunks, build prompt, run LLM."""
    relevant_docs = retriever.invoke(query)
    context = "\n\n".join([d.page_content for d in relevant_docs])

    sources = []
    for d in relevant_docs:
        src  = d.metadata.get("source", "Unknown")
        page = d.metadata.get("page", "?")
        sources.append(f"{os.path.basename(str(src))} · p.{page}")

    prompt = f"""<|system|>
You are a concise legal assistant. Answer ONLY using the context provided.
If the answer is not in the context, respond: "Not available in the provided documents."
Do not add disclaimers or extra commentary.</s>
<|user|>
Context:
{context}

Question: {query}</s>
<|assistant|>
"""
    raw = generator(prompt)[0]["generated_text"]
    answer = raw.split("<|assistant|>")[-1].strip() if "<|assistant|>" in raw else raw[len(prompt):].strip()
    return answer, sources


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚖️ Settings")

    retriever_choice = st.radio(
        "Active retriever",
        ["FAISS (fast)", "Chroma (semantic)", "Both (compare)"],
        index=2,
    )

    st.markdown("---")
    st.markdown("**To add more documents:**")
    st.code(
        "1. Add PDFs to ./legal_docs/\n"
        "2. python index_documents.py\n"
        "3. Restart this app",
        language="bash",
    )
    st.markdown("---")
    st.caption("Answers are based solely on indexed documents. Not legal advice.")


# ──────────────────────────────────────────────
# Main Area
# ──────────────────────────────────────────────
st.markdown('<h1 class="main-title">⚖️ Legal RAG Assistant</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">AI-powered retrieval over your pre-indexed legal document library</p>',
    unsafe_allow_html=True,
)

# ── Check indexes exist ──
faiss_ready  = os.path.exists(FAISS_STORE_PATH)
chroma_ready = os.path.exists(CHROMA_STORE_PATH)

if not faiss_ready or not chroma_ready:
    st.markdown(
        '<div class="error-badge">⚠️ Vector stores not found. '
        'Run <code>python index_documents.py</code> first.</div>',
        unsafe_allow_html=True,
    )
    st.code(
        "# 1. Create folder and place your PDFs inside:\n"
        "mkdir legal_docs\n\n"
        "# 2. Index all PDFs:\n"
        "python index_documents.py\n\n"
        "# 3. Start the app:\n"
        "streamlit run legal_rag_app.py",
        language="bash",
    )
    st.stop()

# ── Load retrievers ──
st.markdown('<div class="ready-badge">✅ Vector stores loaded — ready to query</div>', unsafe_allow_html=True)

with st.spinner("Loading embedding models..."):
    ret_faiss, ret_chroma = load_retrievers()

# ──────────────────────────────────────────────
# Query Interface
# ──────────────────────────────────────────────
st.markdown('<p class="section-header">Ask a Legal Question</p>', unsafe_allow_html=True)

query = st.text_area(
    "Your question",
    placeholder="e.g. What is Section 420 IPC?  |  What constitutes a valid contract?  |  What are fundamental rights?",
    height=90,
)

# Preset queries
preset_queries = [
    "What is Section 420 IPC?",
    "What is theft under IPC?",
    "What is a valid contract?",
    "What are fundamental rights?",
    "What is culpable homicide?",
    "What is anticipatory bail?",
]
st.markdown("**Quick queries:**")
cols = st.columns(3)
for i, pq in enumerate(preset_queries):
    if cols[i % 3].button(pq, key=f"preset_{i}"):
        query = pq

ask_btn = st.button("🔍 Get Legal Answer")

if ask_btn and query.strip():
    with st.spinner("Loading LLM (first run downloads ~600 MB)..."):
        generator = load_llm()

    if retriever_choice in ("FAISS (fast)", "Both (compare)"):
        with st.spinner("Querying FAISS..."):
            ans_f, src_f = rag_pipeline(query, ret_faiss, generator)
        st.markdown('<p class="section-header">🔹 FAISS Answer</p>', unsafe_allow_html=True)
        st.markdown(f'<div class="answer-box">{ans_f}</div>', unsafe_allow_html=True)
        if src_f:
            st.markdown(
                "**Sources:** " + " ".join(f'<span class="source-chip">{s}</span>' for s in src_f),
                unsafe_allow_html=True,
            )

    if retriever_choice in ("Chroma (semantic)", "Both (compare)"):
        with st.spinner("Querying Chroma..."):
            ans_c, src_c = rag_pipeline(query, ret_chroma, generator)
        st.markdown('<p class="section-header">🔸 Chroma Answer</p>', unsafe_allow_html=True)
        st.markdown(f'<div class="answer-box">{ans_c}</div>', unsafe_allow_html=True)
        if src_c:
            st.markdown(
                "**Sources:** " + " ".join(f'<span class="source-chip">{s}</span>' for s in src_c),
                unsafe_allow_html=True,
            )

elif ask_btn:
    st.warning("Please enter a question first.")

st.markdown("---")
st.markdown(
    '<p style="text-align:center;color:#3a3828;font-size:0.8rem;">'
    "Legal RAG Assistant · Answers based solely on indexed documents · Not legal advice</p>",
    unsafe_allow_html=True,
)