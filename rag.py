from loading import load_documents_from_folder
from chunking import chunk_documents
from embeddings import embed_texts
from v_db import get_db_collection
import os
from dotenv import load_dotenv

load_dotenv()

_rag_collection = None

def build_rag_collection():
    """Build RAG collection from sample docs and store embeddings in DB"""
    source_list = load_documents_from_folder("sample_docs/")
    chunks = chunk_documents(source_list)

    ids = [f"chunk_{i}" for i in range(len(chunks))]
    texts = [c["text"] for c in chunks]
    metadatas = [{
        "source": c["source"],
        "doc_id": c["doc_id"],
        "chunk_id": c["chunk_id"]
    } for c in chunks]

    vectors = embed_texts(texts)

    collection = get_db_collection()
    collection.upsert(
        ids=ids,
        embeddings=vectors,
        documents=texts,
        metadatas=metadatas
    )

    return collection

def get_rag_collection():
    """Return singleton RAG collection"""
    global _rag_collection
    if _rag_collection is None:
        _rag_collection = build_rag_collection()
    return _rag_collection
