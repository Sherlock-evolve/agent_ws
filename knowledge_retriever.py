"""Deterministic in-memory semantic retrieval over validated document chunks.

Security boundary: injected production Embeddings implementations may send
document and query text to an external service. Selecting and configuring that
implementation is the caller's responsibility; this module never creates a
network client.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Real

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore


MAX_QUERY_CHARACTERS = 2_000
MAX_SEARCH_RESULTS = 20


class KnowledgeRetrieverError(RuntimeError):
    """Base class for failures that do not disclose indexed or query text."""


class KnowledgeIndexError(KnowledgeRetrieverError):
    """Raised when an injected embedding implementation cannot build an index."""


class KnowledgeSearchError(KnowledgeRetrieverError):
    """Raised when an injected embedding implementation cannot run a search."""


@dataclass(frozen=True)
class RetrievedChunk:
    rank: int
    score: float
    content: str
    source: str
    start_line: int
    end_line: int
    chunk_id: str


class KnowledgeRetriever:
    """A validated, deterministic facade over LangChain's in-memory store."""

    def __init__(
        self,
        chunks: Iterable[Document],
        embeddings: Embeddings,
    ) -> None:
        documents, chunk_ids = self._validate_and_copy_chunks(chunks)
        self.corpus_id = self._calculate_corpus_id(chunk_ids)
        self.chunk_count = len(documents)
        self._vector_store = InMemoryVectorStore(embedding=embeddings)

        if not documents:
            return

        try:
            added_ids = self._vector_store.add_documents(
                documents,
                ids=chunk_ids,
            )
            if added_ids != chunk_ids:
                raise ValueError("embedding result count did not match chunks")
        except Exception:
            raise KnowledgeIndexError(
                "Unable to build the in-memory knowledge index."
            ) from None

    @staticmethod
    def _required_string(
        metadata: dict,
        field: str,
        document_index: int,
    ) -> str:
        value = metadata.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"chunk {document_index} has invalid {field} metadata"
            )
        return value

    @staticmethod
    def _required_line_number(
        metadata: dict,
        field: str,
        document_index: int,
    ) -> int:
        value = metadata.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 1
        ):
            raise ValueError(
                f"chunk {document_index} has invalid {field} metadata"
            )
        return value

    @classmethod
    def _validate_and_copy_chunks(
        cls,
        chunks: Iterable[Document],
    ) -> tuple[list[Document], list[str]]:
        try:
            source_chunks = list(chunks)
        except TypeError:
            raise ValueError("chunks must be an iterable of Documents") from None

        documents: list[Document] = []
        chunk_ids: list[str] = []
        seen_chunk_ids: set[str] = set()

        for document_index, chunk in enumerate(source_chunks):
            if not isinstance(chunk, Document):
                raise ValueError(
                    f"chunk {document_index} is not a Document"
                )
            if (
                not isinstance(chunk.page_content, str)
                or not chunk.page_content.strip()
            ):
                raise ValueError(
                    f"chunk {document_index} has invalid page_content"
                )
            if not isinstance(chunk.metadata, dict):
                raise ValueError(
                    f"chunk {document_index} has invalid metadata"
                )

            chunk_id = cls._required_string(
                chunk.metadata,
                "chunk_id",
                document_index,
            )
            if chunk_id in seen_chunk_ids:
                raise ValueError("chunk_id values must be unique")
            seen_chunk_ids.add(chunk_id)

            source = cls._required_string(
                chunk.metadata,
                "source",
                document_index,
            )
            start_line = cls._required_line_number(
                chunk.metadata,
                "start_line",
                document_index,
            )
            end_line = cls._required_line_number(
                chunk.metadata,
                "end_line",
                document_index,
            )
            if end_line < start_line:
                raise ValueError(
                    f"chunk {document_index} has an invalid line range"
                )

            chunk_ids.append(chunk_id)
            documents.append(
                Document(
                    id=chunk_id,
                    page_content=chunk.page_content,
                    metadata={
                        "source": source,
                        "start_line": start_line,
                        "end_line": end_line,
                        "chunk_id": chunk_id,
                    },
                )
            )

        return documents, chunk_ids

    @staticmethod
    def _calculate_corpus_id(chunk_ids: list[str]) -> str:
        serialized_ids = json.dumps(
            chunk_ids,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(serialized_ids).hexdigest()

    @staticmethod
    def _validate_query(query: str) -> None:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if len(query) > MAX_QUERY_CHARACTERS:
            raise ValueError(
                f"query must not exceed {MAX_QUERY_CHARACTERS} characters"
            )

    @staticmethod
    def _validate_k(k: int) -> None:
        if (
            isinstance(k, bool)
            or not isinstance(k, int)
            or k < 1
            or k > MAX_SEARCH_RESULTS
        ):
            raise ValueError(
                f"k must be an integer from 1 to {MAX_SEARCH_RESULTS}"
            )

    @staticmethod
    def _validate_score_threshold(
        score_threshold: float | None,
    ) -> None:
        if score_threshold is None:
            return
        if (
            isinstance(score_threshold, bool)
            or not isinstance(score_threshold, Real)
            or not math.isfinite(float(score_threshold))
            or not -1 <= score_threshold <= 1
        ):
            raise ValueError(
                "score_threshold must be a finite number from -1 to 1"
            )

    def search(
        self,
        query: str,
        k: int = 4,
        score_threshold: float | None = None,
    ) -> list[RetrievedChunk]:
        self._validate_query(query)
        self._validate_k(k)
        self._validate_score_threshold(score_threshold)

        if self.chunk_count == 0:
            return []

        try:
            candidates = self._vector_store.similarity_search_with_score(
                query,
                k=self.chunk_count,
            )
            validated_candidates = []
            for document, raw_score in candidates:
                if (
                    isinstance(raw_score, bool)
                    or not isinstance(raw_score, Real)
                    or not math.isfinite(float(raw_score))
                ):
                    raise ValueError("vector store returned an invalid score")
                validated_candidates.append(
                    (
                        document,
                        float(raw_score),
                    )
                )
        except Exception:
            raise KnowledgeSearchError(
                "Unable to search the in-memory knowledge index."
            ) from None

        validated_candidates.sort(
            key=lambda item: (
                -item[1],
                item[0].metadata["source"],
                item[0].metadata["start_line"],
                item[0].metadata["chunk_id"],
            )
        )
        if score_threshold is not None:
            validated_candidates = [
                item
                for item in validated_candidates
                if item[1] >= score_threshold
            ]

        return [
            RetrievedChunk(
                rank=rank,
                score=score,
                content=document.page_content,
                source=document.metadata["source"],
                start_line=document.metadata["start_line"],
                end_line=document.metadata["end_line"],
                chunk_id=document.metadata["chunk_id"],
            )
            for rank, (document, score) in enumerate(
                validated_candidates[:k],
                start=1,
            )
        ]


def build_knowledge_retriever(
    chunks: Iterable[Document],
    embeddings: Embeddings,
) -> KnowledgeRetriever:
    """Build an in-memory retriever using caller-selected Embeddings."""

    return KnowledgeRetriever(chunks=chunks, embeddings=embeddings)
