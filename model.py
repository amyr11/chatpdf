from dotenv import load_dotenv
import streamlit as st
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import utils

load_dotenv()


class LangChainRAG:
    def __init__(self, model="llama-3.3-70b-versatile", model_provider="groq"):
        self.model = model
        self.model_provider = model_provider
        self.docs = None
        self.chunked_docs = None
        self.vectorstore = None

        self.llm = init_chat_model(self.model, model_provider=self.model_provider)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )

    def get_docs(self, files):
        self.docs = utils.load_pdfs(files)
        return self.docs

    def chunk(self, docs, chunk_size=1000, chunk_overlap=250):
        self.chunked_docs = utils.chunk(docs, chunk_size, chunk_overlap)
        return self.chunked_docs

    def embed_store(self, docs):
        self.vectorstore = utils.get_vectorstore
        return self.vectorstore

    def clear_docs(self):
        self.docs = None
        self.chunked_docs = None
        self.vectorstore = None
