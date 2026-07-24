from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from knowledge_retriever import (
    KnowledgeIndexError,
    KnowledgeRetriever,
    KnowledgeSearchError,
    RetrievedChunk,
)


class ControlledEmbeddings(Embeddings):
    def __init__(
        self,
        document_vectors=None,
        query_vectors=None,
        *,
        document_error=None,
        query_error=None,
    ):
        self.document_vectors = document_vectors or {}
        self.query_vectors = query_vectors or {}
        self.document_error = document_error
        self.query_error = query_error
        self.document_calls = []
        self.query_calls = []

    def embed_documents(self, texts):
        self.document_calls.append(list(texts))
        if self.document_error is not None:
            raise self.document_error
        return [self.document_vectors[text] for text in texts]

    def embed_query(self, text):
        self.query_calls.append(text)
        if self.query_error is not None:
            raise self.query_error
        return self.query_vectors[text]


def make_chunk(
    chunk_id,
    content,
    source,
    start_line,
    end_line=None,
):
    return Document(
        page_content=content,
        metadata={
            "chunk_id": chunk_id,
            "source": source,
            "start_line": start_line,
            "end_line": end_line or start_line,
        },
    )


def test_controlled_vectors_produce_expected_ranking_and_scores():
    chunks = [
        make_chunk("alpha-id", "alpha", "docs/a.txt", 1),
        make_chunk("beta-id", "beta", "docs/b.txt", 2),
        make_chunk("opposite-id", "opposite", "docs/c.txt", 3),
    ]
    embeddings = ControlledEmbeddings(
        document_vectors={
            "alpha": [1.0, 0.0],
            "beta": [0.8, 0.6],
            "opposite": [-1.0, 0.0],
        },
        query_vectors={"topic": [1.0, 0.0]},
    )
    retriever = KnowledgeRetriever(chunks, embeddings)

    results = retriever.search("topic", k=3)
    thresholded = retriever.search(
        "topic",
        k=3,
        score_threshold=0.0,
    )

    assert [result.chunk_id for result in results] == [
        "alpha-id",
        "beta-id",
        "opposite-id",
    ]
    assert [result.rank for result in results] == [1, 2, 3]
    assert [result.score for result in results] == pytest.approx(
        [1.0, 0.8, -1.0]
    )
    assert [result.chunk_id for result in thresholded] == [
        "alpha-id",
        "beta-id",
    ]
    expected_corpus_id = hashlib.sha256(
        json.dumps(
            ["alpha-id", "beta-id", "opposite-id"],
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert retriever.corpus_id == expected_corpus_id
    assert list(retriever._vector_store.store) == [
        "alpha-id",
        "beta-id",
        "opposite-id",
    ]


def test_equal_scores_use_stable_metadata_tie_breakers_before_k():
    chunks = [
        make_chunk("z-id", "z", "docs/z.txt", 1),
        make_chunk("line-two", "line two", "docs/a.txt", 2),
        make_chunk("c-id", "c", "docs/a.txt", 1),
        make_chunk("a-id", "a", "docs/a.txt", 1),
    ]
    embeddings = ControlledEmbeddings(
        document_vectors={
            chunk.page_content: [1.0, 0.0]
            for chunk in chunks
        },
        query_vectors={"same": [1.0, 0.0]},
    )
    retriever = KnowledgeRetriever(chunks, embeddings)

    first = retriever.search("same", k=2)
    second = retriever.search("same", k=2)

    assert [result.chunk_id for result in first] == ["a-id", "c-id"]
    assert first == second


def test_results_preserve_citations_are_frozen_and_do_not_alias_documents():
    original = make_chunk(
        "stable-id",
        "original content",
        "docs/reference.md",
        7,
        9,
    )
    embeddings = ControlledEmbeddings(
        document_vectors={"original content": [1.0]},
        query_vectors={"reference": [1.0]},
    )
    retriever = KnowledgeRetriever([original], embeddings)

    assert original.id is None
    original.page_content = "mutated content"
    original.metadata["source"] = "docs/mutated.txt"
    result = retriever.search("reference", k=1)[0]

    assert isinstance(result, RetrievedChunk)
    assert result.content == "original content"
    assert result.source == "docs/reference.md"
    assert result.start_line == 7
    assert result.end_line == 9
    assert result.chunk_id == "stable-id"
    with pytest.raises(FrozenInstanceError):
        result.source = "docs/changed.txt"


def test_invalid_queries_duplicate_ids_and_broken_metadata_are_rejected():
    valid = make_chunk("same-id", "valid", "docs/a.txt", 1)
    embeddings = ControlledEmbeddings(
        document_vectors={"valid": [1.0], "other": [1.0]},
        query_vectors={"query": [1.0]},
    )

    with pytest.raises(ValueError, match="unique"):
        KnowledgeRetriever(
            [
                valid,
                make_chunk("same-id", "other", "docs/b.txt", 1),
            ],
            embeddings,
        )

    broken_chunks = [
        Document(
            page_content="",
            metadata={
                "chunk_id": "empty",
                "source": "docs/a.txt",
                "start_line": 1,
                "end_line": 1,
            },
        ),
        Document(
            page_content="content",
            metadata={
                "chunk_id": "missing-source",
                "start_line": 1,
                "end_line": 1,
            },
        ),
        make_chunk("bad-start", "content", "docs/a.txt", 0),
        make_chunk("bad-range", "content", "docs/a.txt", 3, 2),
    ]
    for broken_chunk in broken_chunks:
        with pytest.raises(ValueError):
            KnowledgeRetriever([broken_chunk], embeddings)

    retriever = KnowledgeRetriever([valid], embeddings)
    invalid_searches = [
        {"query": ""},
        {"query": "   "},
        {"query": "x" * 2_001},
        {"query": "query", "k": 0},
        {"query": "query", "k": 21},
        {"query": "query", "k": True},
        {"query": "query", "k": 1.5},
        {"query": "query", "score_threshold": float("nan")},
        {"query": "query", "score_threshold": float("inf")},
        {"query": "query", "score_threshold": -1.1},
        {"query": "query", "score_threshold": 1.1},
        {"query": "query", "score_threshold": True},
    ]
    for arguments in invalid_searches:
        with pytest.raises(ValueError):
            retriever.search(**arguments)


def test_empty_corpus_and_embedding_failures_are_safe():
    empty_embeddings = ControlledEmbeddings()
    empty_retriever = KnowledgeRetriever([], empty_embeddings)

    assert empty_retriever.search("anything") == []
    assert empty_embeddings.document_calls == []
    assert empty_embeddings.query_calls == []

    chunk = make_chunk(
        "secret-id",
        "document-secret-sentinel",
        "docs/a.txt",
        1,
    )
    build_embeddings = ControlledEmbeddings(
        document_error=RuntimeError("credential-secret-sentinel"),
    )
    with pytest.raises(KnowledgeIndexError) as build_error:
        KnowledgeRetriever([chunk], build_embeddings)
    assert "document-secret-sentinel" not in str(build_error.value)
    assert "credential-secret-sentinel" not in str(build_error.value)

    query_embeddings = ControlledEmbeddings(
        document_vectors={"document-secret-sentinel": [1.0]},
        query_error=RuntimeError("api-key-secret-sentinel"),
    )
    retriever = KnowledgeRetriever([chunk], query_embeddings)
    with pytest.raises(KnowledgeSearchError) as query_error:
        retriever.search("query-secret-sentinel")
    assert "query-secret-sentinel" not in str(query_error.value)
    assert "api-key-secret-sentinel" not in str(query_error.value)
