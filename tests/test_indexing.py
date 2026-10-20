"""FAISS indexing through ChatIngestor / FaissManager, including de-duplication."""
import pytest

from src.document_ingestion.data_ingestion import ChatIngestor, FaissManager
from tests.helpers import FakeUpload
from utils.model_loader import ModelLoader

PARAGRAPHS = [f"Paragraph {i}: the {word} report covers quarter {i} results in detail."
              for i, word in enumerate(["finance", "marketing", "legal", "hiring", "travel", "security"])]


def write_doc(path, paragraphs):
    path.write_text("\n\n".join(paragraphs), encoding="utf-8")
    return path


def index_size(ingestor):
    fm = FaissManager(ingestor.faiss_dir, ModelLoader())
    return len(fm.load_or_create().index_to_docstore_id)


@pytest.fixture
def ingestor(workdir, fake_models):
    return ChatIngestor(temp_base="data", faiss_base="faiss_index", session_id="indexing")


def test_fresh_index_has_no_duplicate_chunks(ingestor, workdir):
    # Regression: a new index stored every chunk twice
    doc = write_doc(workdir / "a.txt", PARAGRAPHS)
    ingestor.build_retriever([FakeUpload(doc)], chunk_size=80, chunk_overlap=0)

    assert ingestor.stats["chunks"] > 1
    assert index_size(ingestor) == ingestor.stats["chunks"] == ingestor.stats["added"]


def test_reuploading_the_same_document_adds_nothing(ingestor, workdir):
    doc = write_doc(workdir / "a.txt", PARAGRAPHS)
    ingestor.build_retriever([FakeUpload(doc)], chunk_size=80, chunk_overlap=0)
    first = index_size(ingestor)

    ingestor.build_retriever([FakeUpload(doc)], chunk_size=80, chunk_overlap=0)

    assert ingestor.stats["added"] == 0
    assert index_size(ingestor) == first


def test_adding_a_second_file_indexes_all_of_its_chunks(ingestor, workdir):
    # Regression: chunks were keyed by source path only, so a new file added to an
    # existing session contributed just its first chunk
    first = write_doc(workdir / "a.txt", PARAGRAPHS[:3])
    second = write_doc(workdir / "b.txt", PARAGRAPHS[3:])
    ingestor.build_retriever([FakeUpload(first)], chunk_size=80, chunk_overlap=0)
    before = index_size(ingestor)

    ingestor.build_retriever([FakeUpload(second)], chunk_size=80, chunk_overlap=0)

    assert ingestor.stats["added"] == ingestor.stats["chunks"] > 1
    assert index_size(ingestor) == before + ingestor.stats["chunks"]


def test_sessions_are_isolated(workdir, fake_models):
    doc = write_doc(workdir / "a.txt", PARAGRAPHS)
    one = ChatIngestor(temp_base="data", faiss_base="faiss_index", session_id="one")
    two = ChatIngestor(temp_base="data", faiss_base="faiss_index", session_id="two")

    one.build_retriever([FakeUpload(doc)], chunk_size=80, chunk_overlap=0)

    assert (workdir / "faiss_index" / "one" / "index.faiss").exists()
    assert not (workdir / "faiss_index" / "two" / "index.faiss").exists()
