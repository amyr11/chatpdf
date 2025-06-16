import streamlit as st
import uuid
import utils
import time
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool, InjectedToolArg
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph
from langgraph.graph import END
from langgraph.prebuilt import tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableConfig

load_dotenv()


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

st.title("ChatPDF 🤖 📑")

st.session_state.setdefault("last_uploaded_files", [])
st.session_state.setdefault("used_files", [])
# st.session_state.setdefault("vector_store", None)
st.session_state.setdefault("graph", None)

# if "vector_store" not in st.session_state:
#     st.session_state.vector_store = None

# Initialize LLM
if "llm" not in st.session_state:
    # llm = init_chat_model("llama-3.3-70b-versatile", model_provider="groq")
    llm = ChatGroq(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        # other params...
    )
    st.session_state.setdefault("llm", llm)


# Session id
if "chat_session_id" not in st.session_state:
    st.session_state.chat_session_id = str(uuid.uuid4())

# Set config
if "config" not in st.session_state:
    st.session_state.setdefault(
        "config", {"configurable": {"thread_id": st.session_state.chat_session_id}}
    )

st.session_state.setdefault(st.session_state.chat_session_id, {})

st.write(st.session_state.chat_session_id)


def chat_message(name):
    return st.container(key=f"{name}-{uuid.uuid4()}").chat_message(name=name)


def upload_files():
    for file in st.session_state.used_files:
        print("FILE,", file.name)
    all_docs = utils.load_pdfs(st.session_state.used_files)
    st.session_state.all_docs = all_docs
    return all_docs


def chunk():
    docs_chunked = utils.chunk(
        st.session_state.all_docs,
        st.session_state.used_chunk_size,
        st.session_state.used_chunk_overlap,
    )
    st.session_state.docs_chunked = docs_chunked
    time.sleep(1)
    return docs_chunked


def embed():
    if st.session_state[st.session_state.chat_session_id].get("vector_store"):
        st.session_state[st.session_state.chat_session_id][
            "vector_store"
        ].reset_collection()

    vector_store = utils.get_vectorstore(
        st.session_state.docs_chunked, st.session_state.chat_session_id
    )
    # st.session_state.vector_store = vector_store
    st.session_state[st.session_state.chat_session_id]["vector_store"] = vector_store
    st.session_state.config["configurable"]["vector_store"] = st.session_state[
        st.session_state.chat_session_id
    ]["vector_store"]
    print("NEW VECTOR STORE")
    return vector_store


rerun = False

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
    st.session_state.setdefault("used_k", None)
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
            st.session_state[st.session_state.chat_session_id]["vector_store"] = None

    if rerun:
        st.rerun()
        rerun = False

    if st.session_state.last_uploaded_files:
        files_changed = False

        # Extract texts and create documents
        if st.session_state.used_files != st.session_state.last_uploaded_files:
            with st.spinner("Uploading...", show_time=True):
                files_changed = True
                st.cache_data.clear()
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
                vector_store = embed()
                if vector_store:
                    st.success(f"Embeddings stored in Chroma")

            # used_vector_store = st.session_state[st.session_state.chat_session_id].get(
            #     "vector_store"
            # )

            def get_vector_store():
                return st.session_state[st.session_state.chat_session_id].get(
                    "vector_store"
                )

            # Create graphxj
            @tool(
                response_format="content_and_artifact",
                description="Retrieve information related to a query.",
            )
            def retrieve(query: str, config: RunnableConfig):
                vs = config["configurable"].get("vector_store")
                retrieved_docs = vs.similarity_search(
                    query, k=4  # TODO: replace with session state
                )
                serialized = "\n\n".join(
                    (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
                    for doc in retrieved_docs
                )
                return serialized, retrieved_docs

            def query_or_respond(state: MessagesState):
                """Generate tool call for retrieval or respond."""
                llm_with_tools = st.session_state.llm.bind_tools([retrieve])
                response = llm_with_tools.invoke(state["messages"])
                return {"messages": [response]}

            tools = ToolNode([retrieve])

            def generate(state: MessagesState):
                """Generate answer."""
                recent_tool_messages = []
                # Get the most recent tool messages
                for message in reversed(state["messages"]):
                    if message.type == "tool":
                        recent_tool_messages.append(message)
                    else:
                        break
                # Reverse the list to preserve chronological order
                tool_messages = recent_tool_messages[::-1]

                docs_content = "\n\n".join(doc.content for doc in tool_messages)
                system_message_content = (
                    "Your name is ChatPDF, you are created by Amyr. "
                    "You are an assistant for question-answering tasks. "
                    "Use the following pieces of retrieved context to answer "
                    "the question. If you don't know the answer, say that you "
                    "don't know. Use three sentences maximum and keep the "
                    "answer concise."
                    "\n\n"
                    f"{docs_content}"
                )
                conversation_messages = [
                    message
                    for message in state["messages"]
                    if message.type in ("human", "system")
                    or (message.type == "ai" and not message.tool_calls)
                ]
                prompt = [SystemMessage(system_message_content)] + conversation_messages

                response = st.session_state.llm.invoke(prompt)
                return {"messages": [response]}

            graph_builder = StateGraph(MessagesState)

            graph_builder.add_node(query_or_respond)
            graph_builder.add_node(tools)
            graph_builder.add_node(generate)

            graph_builder.set_entry_point("query_or_respond")
            graph_builder.add_conditional_edges(
                "query_or_respond", tools_condition, {END: END, "tools": "tools"}
            )
            graph_builder.add_edge("tools", "generate")
            graph_builder.add_edge("generate", END)

            memory = MemorySaver()
            graph = graph_builder.compile(checkpointer=memory)
            st.session_state.graph = graph
            print("NEW GRAPH")

        files_changed = False


def chat(query):
    graph = st.session_state.graph
    input_message = [HumanMessage(query)]
    print("DEBUG", st.session_state.config)
    response = graph.invoke({"messages": input_message}, st.session_state.config)
    response_content = response["messages"][-1].content
    return response_content


# Display filenames of uploaded files
badges = ""
if st.session_state[st.session_state.chat_session_id].get("vector_store"):
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
    if st.session_state[st.session_state.chat_session_id].get("vector_store"):
        response = chat(prompt)
    else:
        response = "Please upload a PDF first"
    # Display chatbot response
    with chat_message("assistant"):
        st.markdown(response)
    # Add chatbot response to history
    st.session_state.messages.append({"role": "assistant", "content": response})
