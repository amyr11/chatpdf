import streamlit as st
import uuid
import utils
import time

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


def chat_message(name):
    return st.container(key=f"{name}-{uuid.uuid4()}").chat_message(name=name)


docs_processed = False

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

    st.divider()

    # File uploader
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        accept_multiple_files=True,
        type="pdf",
    )

    if uploaded_files:
        # Extract texts and create documents
        with st.spinner("Uploading...", show_time=True):
            all_docs = utils.load_pdfs(uploaded_files)
            st.success(f"{len(all_docs)} file(s) uploaded")

        # Chunk documents
        with st.spinner("Chunking documents...", show_time=True):
            docs_chunked = utils.chunk(all_docs)
            time.sleep(1)
            st.success(f"Documents chunked into {len(docs_chunked)} documents")

        # Embed and store in Chroma
        with st.spinner("Calculating embeddings...", show_time=True):
            vector_store = utils.get_vectorstore(docs_chunked)
            docs_processed = True
            st.success(f"Embeddings stored in Chroma")
    else:
        docs_processed = False

st.title("ChatPDF 🤖 📑")

# Display filenames of uploaded files
badges = ""
if docs_processed:
    for file in uploaded_files:
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
