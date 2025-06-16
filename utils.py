__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")


def load_pdfs(files):
    """
    Args:
        file_paths: List of file paths to PDF documents to be loaded.

    Returns:
        all_docs: A list of loaded document objects from the provided PDF files.
    """
    from langchain_core.documents import Document
    from pypdf import PdfReader

    all_docs = []

    def clean_text(text):
        return " ".join(text.split())

    for file in files:
        filename = file.name
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += clean_text(page.extract_text())
        doc = Document(
            id=filename,
            page_content=text,
            metadata={"source": filename},
        )
        all_docs.append(doc)

    return all_docs


def chunk(docs, chunk_size=1000, chunk_overlap=250):
    """
    Splits a list of documents into smaller chunks.

    Args:
        docs: List of Document objects to be chunked.
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        docs_chunked: List of chunked Document objects.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    docs_chunked = text_splitter.split_documents(docs)

    return docs_chunked


def get_vectorstore(docs, collection_name):
    """
    Creates a vector store from a list of documents.

    Args:
        docs: List of Document objects to be embedded and stored.

    Returns:
        vector_store: A Chroma vector store containing the embedded documents.
    """
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    vector_store = Chroma.from_documents(
        docs, embeddings, collection_name=collection_name
    )

    return vector_store
