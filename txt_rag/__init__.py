"""txt_rag: 纯文本目录 RAG（检索增强生成）CLI。"""
from .index import TextIndex, Chunk
from .retriever import Retriever

__all__ = ["TextIndex", "Chunk", "Retriever"]
__version__ = "0.1.0"
