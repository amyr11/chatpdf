from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
import utils
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import MessagesState
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.graph import END
from langgraph.prebuilt import tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage

load_dotenv()


class LangChainRAG:
    def __init__(
        self,
        thread_id,
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        model_provider="groq",
    ):
        self.thread_id = thread_id
        self.model = model
        self.model_provider = model_provider
        self.docs = None
        self.chunked_docs = None
        self.vector_store = None
        self.graph = None
        self.retrieval_k = 4

        self.llm = init_chat_model(self.model, model_provider=self.model_provider)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )

    def get_response(self, query):
        config = {
            "configurable": {
                "thread_id": self.thread_id,
                "vector_store": self.vector_store,
                "retrieval_k": self.retrieval_k,
            },
        }

        input_message = [HumanMessage(query)]
        response = self.graph.invoke(
            {"messages": input_message},
            config,
        )

        return response

    def parse_ai_message(self, response):
        return response["messages"][-1].content

    def parse_tool_call(self, response):
        latest_tool_message = next(
            (m for m in reversed(response["messages"]) if m.type == "tool"), None
        )

        if latest_tool_message:
            return latest_tool_message.content

        return ""

    def get_graph(self):
        @tool(
            response_format="content_and_artifact",
            description="Retrieve information related to a query.",
        )
        def retrieve(query: str, config: RunnableConfig):
            vector_store = config["configurable"].get("vector_store")
            retrieved_docs = vector_store.similarity_search(query, k=self.retrieval_k)
            serialized = "\n\n----------\n\n".join(
                (f"Source: {doc.metadata["source"]}\nContent: {doc.page_content}")
                for doc in retrieved_docs
            )
            return serialized, retrieved_docs

        def query_or_respond(state: MessagesState):
            """Generate tool call for retrieval or respond."""
            llm_with_tools = self.llm.bind_tools([retrieve])
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

            response = self.llm.invoke(prompt)
            return {"messages": [response]}

        # Build graph
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
        self.graph = graph_builder.compile(checkpointer=memory)

        return self.graph

    def get_docs(self, files):
        self.docs = utils.load_pdfs(files)
        return self.docs

    def chunk(self, docs, chunk_size=1000, chunk_overlap=250):
        self.chunked_docs = utils.chunk(docs, chunk_size, chunk_overlap)
        return self.chunked_docs

    def embed_store(self, docs):
        # Reset collection if exists
        if self.vector_store:
            self.vector_store.reset_collection()
        self.vector_store = utils.get_vectorstore(docs, self.thread_id)
        return self.vector_store

    def clear_docs(self):
        self.docs = None
        self.chunked_docs = None
        self.vector_store = None
