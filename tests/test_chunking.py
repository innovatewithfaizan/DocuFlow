"""Chunking: RecursiveCharacterTextSplitter as configured by ChatIngestor._split."""
import pytest
from langchain_core.documents import Document

from src.document_ingestion.data_ingestion import ChatIngestor

SENTENCES = [f"Sentence number {i} describes item {i} in some detail." for i in range(120)]
LONG_TEXT = " ".join(SENTENCES)


@pytest.fixture
def ingestor(workdir, fake_models):
    return ChatIngestor(temp_base="data", faiss_base="faiss_index", session_id="chunking")


def test_chunks_respect_chunk_size(ingestor):
    chunks = ingestor._split([Document(page_content=LONG_TEXT)], chunk_size=300, chunk_overlap=50)

    assert len(chunks) > 1
    assert all(len(c.page_content) <= 300 for c in chunks)


def test_consecutive_chunks_overlap(ingestor):
    chunks = ingestor._split([Document(page_content=LONG_TEXT)], chunk_size=300, chunk_overlap=100)

    for prev, nxt in zip(chunks, chunks[1:]):
        # the start of each chunk repeats text from the end of the previous one
        head = nxt.page_content[:30]
        assert head in prev.page_content


def test_no_text_is_lost(ingestor):
    chunks = ingestor._split([Document(page_content=LONG_TEXT)], chunk_size=250, chunk_overlap=0)
    joined = " ".join(c.page_content for c in chunks)

    for sentence in SENTENCES:
        assert sentence in joined


def test_metadata_is_carried_to_every_chunk(ingestor):
    doc = Document(page_content=LONG_TEXT, metadata={"source": "report.pdf", "page": 4})
    chunks = ingestor._split([doc], chunk_size=300, chunk_overlap=50)

    assert all(c.metadata == {"source": "report.pdf", "page": 4} for c in chunks)


def test_smaller_chunks_mean_more_chunks(ingestor):
    big = ingestor._split([Document(page_content=LONG_TEXT)], chunk_size=1000, chunk_overlap=0)
    small = ingestor._split([Document(page_content=LONG_TEXT)], chunk_size=200, chunk_overlap=0)

    assert len(small) > len(big)
