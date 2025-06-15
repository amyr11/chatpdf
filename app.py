import streamlit as st
import uuid
import time
from model import LangChainRAG

st.set_page_config("ChatPDF", "📑")
st.html(
    """
    <style>
    [class*="st-key-user"] {
        background-color: #e3e2da;
        border-radius: 20px;
    }

    </style>
    """
)

if "rag_model" not in st.session_state:
    st.session_state.rag_model = LangChainRAG()

st.session_state.setdefault("last_uploaded_files", [])
st.session_state.setdefault("used_files", [])

rerun = False


def chat_message(name):
    return st.container(key=f"{name}-{uuid.uuid4()}").chat_message(name=name)


def upload_files():
    all_docs = st.session_state.rag_model.get_docs(st.session_state.used_files)
    st.session_state.all_docs = all_docs
    return all_docs


def chunk():
    docs_chunked = st.session_state.rag_model.chunk(
        st.session_state.all_docs,
        st.session_state.used_chunk_size,
        st.session_state.used_chunk_overlap,
    )
    st.session_state.docs_chunked = docs_chunked
    time.sleep(1)
    return docs_chunked


def embed_store():
    vector_store = st.session_state.rag_model.embed_store(st.session_state.docs_chunked)
    st.session_state.vector_store = vector_store
    time.sleep(1)
    return vector_store


# Sidebar
with st.sidebar:
    # Model parameters
    st.write("Model parameters")
    retriever_k = st.slider(
        "No. of retrieved docs",
        1,
        8,
        4,
        help="Determines how many top text chunks will be retrieved based on similarity to the user's query. Higher values increase recall but may include less relevant results.",
    )
    splitter_chunk_size = st.slider(
        "Chunk size",
        1,
        2000,
        1000,
        help="Determines the size of each split.",
    )
    splitter_chunk_overlap = st.slider(
        "Chunk overlap",
        1,
        splitter_chunk_size - 1,
        splitter_chunk_size // 4,
        help="Determines the size of each split.",
    )

    st.session_state.setdefault("last_k", retriever_k)
    st.session_state.setdefault("last_chunk_size", splitter_chunk_size)
    st.session_state.setdefault("used_chunk_size", None)
    st.session_state.setdefault("last_chunk_overlap", splitter_chunk_overlap)
    st.session_state.setdefault("used_chunk_overlap", None)

    if (
        st.session_state.last_k != retriever_k
        or st.session_state.last_chunk_size != splitter_chunk_size
        or st.session_state.last_chunk_overlap != splitter_chunk_overlap
    ):
        if st.button("Save changes"):
            st.session_state.last_k = retriever_k
            st.session_state.last_chunk_size = splitter_chunk_size
            st.session_state.last_chunk_overlap = splitter_chunk_overlap
            rerun = True

    st.divider()

    # File uploader
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        accept_multiple_files=True,
        type="pdf",
    )

    if st.session_state.last_uploaded_files != uploaded_files:
        st.session_state.last_uploaded_files = uploaded_files

        if not st.session_state.last_uploaded_files:
            st.session_state.used_files = []

    if rerun:
        print("RERUNNING")
        st.rerun()
        rerun = False

    if st.session_state.last_uploaded_files:
        files_changed = False

        # Extract texts and create documents
        if st.session_state.used_files != st.session_state.last_uploaded_files:
            with st.spinner("Uploading...", show_time=True):
                files_changed = True
                st.session_state.used_files = st.session_state.last_uploaded_files
                all_docs = upload_files()
                st.success(f"{len(all_docs)} file(s) uploaded")

        if (
            files_changed
            or st.session_state.used_chunk_size != st.session_state.last_chunk_size
            or st.session_state.used_chunk_overlap
            != st.session_state.last_chunk_overlap
        ):
            # Chunk documents
            with st.spinner("Chunking documents...", show_time=True):
                st.session_state.used_chunk_size = st.session_state.last_chunk_size
                st.session_state.used_chunk_overlap = (
                    st.session_state.last_chunk_overlap
                )
                docs_chunked = chunk()
                st.success(
                    f"Documents chunked into {len(docs_chunked)} documents using chunk size: {splitter_chunk_size} and chunk overlap: {splitter_chunk_overlap}"
                )

            # Embed and store in Chroma
            with st.spinner("Calculating embeddings...", show_time=True):
                vector_store = embed_store()
                st.success(f"Embeddings stored in Chroma")

        files_changed = False

st.title("ChatPDF 🤖 📑")

# Display filenames of uploaded files
badges = ""
if st.session_state.used_files:
    for file in st.session_state.used_files:
        badge = ":green-badge[:material/check: {title}]"
        badges += badge.format(title=file.name) + " "
    st.markdown(badges)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display messages
for message in st.session_state.messages:
    with chat_message(message["role"]):
        st.markdown(message["content"])

# Input message
if prompt := st.chat_input("Ask ChatPDF"):
    # Display user message
    with chat_message("user"):
        st.markdown(prompt)
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Chatbot response
    response = f"Echo: {prompt}"
    # Display chatbot response
    with chat_message("assistant"):
        st.markdown(response)
    # Add chatbot response to history
    st.session_state.messages.append({"role": "assistant", "content": response})
