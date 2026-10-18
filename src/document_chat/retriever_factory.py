"""
Builds the retriever used by every chat path, from config/config.yaml (`retriever:` block)
plus optional per-request overrides.

    similarity   plain top-k cosine search over the FAISS index
    mmr          Maximal Marginal Relevance: fetch `fetch_k` candidates, then pick `k`
                 that are relevant but not redundant with each other (lambda_mult
                 trades relevance for diversity)
    compression  optional post-processing: split the retrieved chunks into smaller
                 pieces, drop near-duplicate pieces and keep the `keep_top_n` pieces
                 most similar to the question, so the LLM sees less irrelevant text
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline, EmbeddingsFilter
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.config_loader import load_config

SEARCH_TYPES = {"similarity", "mmr"}


@dataclass(frozen=True)
class CompressionSettings:
    enabled: bool = False
    piece_size: int = 400
    keep_top_n: int = 8
    redundancy_threshold: float = 0.95


@dataclass(frozen=True)
class RetrieverSettings:
    search_type: str = "similarity"
    k: int = 5
    fetch_k: int = 20
    lambda_mult: float = 0.5
    compression: CompressionSettings = field(default_factory=CompressionSettings)

    @classmethod
    def from_config(cls, config: Optional[Dict[str, Any]] = None, **overrides: Any) -> "RetrieverSettings":
        """Read the `retriever:` block of config.yaml, then apply non-None overrides."""
        block = (config if config is not None else load_config()).get("retriever", {}) or {}
        comp_block = block.get("compression", {}) or {}
        settings = cls(
            search_type=block.get("search_type", cls.search_type),
            k=int(block.get("top_k", cls.k)),
            fetch_k=int(block.get("fetch_k", cls.fetch_k)),
            lambda_mult=float(block.get("lambda_mult", cls.lambda_mult)),
            compression=CompressionSettings(
                enabled=bool(comp_block.get("enabled", False)),
                piece_size=int(comp_block.get("piece_size", CompressionSettings.piece_size)),
                keep_top_n=int(comp_block.get("keep_top_n", CompressionSettings.keep_top_n)),
                redundancy_threshold=float(
                    comp_block.get("redundancy_threshold", CompressionSettings.redundancy_threshold)
                ),
            ),
        )
        return settings.with_overrides(**overrides)

    def with_overrides(self, compress: Optional[bool] = None, **overrides: Any) -> "RetrieverSettings":
        settings = replace(self, **{k: v for k, v in overrides.items() if v is not None})
        if compress is not None:
            settings = replace(settings, compression=replace(settings.compression, enabled=compress))
        return settings


def build_retriever(
    vectorstore: VectorStore,
    settings: RetrieverSettings,
    embeddings: Optional[Embeddings] = None,
) -> BaseRetriever:
    """Wrap a vector store in a similarity/MMR retriever, optionally with compression."""
    if settings.search_type not in SEARCH_TYPES:
        raise ValueError(f"search_type must be one of {sorted(SEARCH_TYPES)}, got {settings.search_type!r}")
    if settings.k < 1:
        raise ValueError("k must be >= 1")

    search_kwargs: Dict[str, Any] = {"k": settings.k}
    if settings.search_type == "mmr":
        search_kwargs["fetch_k"] = max(settings.fetch_k, settings.k)
        search_kwargs["lambda_mult"] = settings.lambda_mult

    base = vectorstore.as_retriever(search_type=settings.search_type, search_kwargs=search_kwargs)
    if not settings.compression.enabled:
        return base

    embeddings = embeddings or getattr(vectorstore, "embeddings", None) or getattr(vectorstore, "embedding_function", None)
    if embeddings is None:
        raise ValueError("Compression needs an embeddings model")

    comp = settings.compression
    pipeline = DocumentCompressorPipeline(
        transformers=[
            RecursiveCharacterTextSplitter(chunk_size=comp.piece_size, chunk_overlap=0),
            EmbeddingsRedundantFilter(embeddings=embeddings, similarity_threshold=comp.redundancy_threshold),
            EmbeddingsFilter(embeddings=embeddings, k=comp.keep_top_n),
        ]
    )
    return ContextualCompressionRetriever(base_compressor=pipeline, base_retriever=base)
