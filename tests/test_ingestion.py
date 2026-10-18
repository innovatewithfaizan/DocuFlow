"""Loading and saving documents: the first stage of the pipeline."""
import pytest

from exception.custom_exception import DocuFlowException
from src.document_ingestion.data_ingestion import ChatIngestor, DocHandler, DocumentComparator
from utils.document_ops import load_documents
from utils.file_io import save_uploaded_files, validate_session_id
from tests.helpers import FakeUpload, make_docx, make_pdf


def test_load_documents_reads_pdf_txt_and_docx(tmp_path):
    pdf = make_pdf(tmp_path / "report.pdf", ["Page one talks about revenue.", "Page two talks about costs."])
    txt = tmp_path / "notes.txt"
    txt.write_text("Plain text notes about hiring.", encoding="utf-8")
    docx = make_docx(tmp_path / "memo.docx", ["A memo about the office move."])

    docs = load_documents([pdf, txt, docx])

    # PyPDFLoader yields one Document per page and keeps page numbers
    pdf_docs = [d for d in docs if d.metadata["source"].endswith("report.pdf")]
    assert [d.metadata["page"] for d in pdf_docs] == [0, 1]
    assert "revenue" in pdf_docs[0].page_content and "costs" in pdf_docs[1].page_content

    texts = " ".join(d.page_content for d in docs)
    assert "hiring" in texts
    assert "office move" in texts


def test_save_uploaded_files_skips_unsupported_types(tmp_path):
    good = tmp_path / "a.txt"
    good.write_text("hello", encoding="utf-8")
    bad = tmp_path / "b.exe"
    bad.write_bytes(b"MZ")

    saved = save_uploaded_files([FakeUpload(good), FakeUpload(bad)], tmp_path / "out")

    assert len(saved) == 1
    assert saved[0].suffix == ".txt"
    assert saved[0].read_text(encoding="utf-8") == "hello"


def test_doc_handler_reads_every_page(workdir):
    # Regression: read_pdf used to return after the first page
    pdf = make_pdf(workdir / "three.pdf", ["alpha page", "bravo page", "charlie page"])
    handler = DocHandler(data_dir=str(workdir / "analysis"))

    saved = handler.save_pdf(FakeUpload(pdf))
    text = handler.read_pdf(saved)

    for marker in ("alpha", "bravo", "charlie"):
        assert marker in text
    assert "--- Page 3 ---" in text


def test_comparator_combines_pages_with_real_newlines(workdir):
    # Regression: pages were joined with the literal string "/n"
    ref = make_pdf(workdir / "ref.pdf", ["first page", "second page"])
    act = make_pdf(workdir / "act.pdf", ["first page", "second page changed"])
    comp = DocumentComparator(base_dir=str(workdir / "compare"))

    comp.save_uploaded_files(FakeUpload(ref), FakeUpload(act))
    combined = comp.combine_documents()

    assert "/n" not in combined
    assert "Document: act.pdf" in combined and "Document: ref.pdf" in combined


def test_comparator_strips_directories_from_upload_names(workdir):
    pdf = make_pdf(workdir / "x.pdf", ["content"])
    comp = DocumentComparator(base_dir=str(workdir / "compare"))

    ref_path, act_path = comp.save_uploaded_files(
        FakeUpload(pdf, name="../../escape.pdf"), FakeUpload(pdf, name="nested/dir/act.pdf")
    )

    assert ref_path.parent == comp.session_path and ref_path.name == "escape.pdf"
    assert act_path.parent == comp.session_path and act_path.name == "act.pdf"


@pytest.mark.parametrize("session_id", ["session_20250101_120000_abcd1234", "abc", "A-1_b"])
def test_valid_session_ids(session_id):
    assert validate_session_id(session_id) == session_id


@pytest.mark.parametrize("session_id", ["", "../etc", "a/b", "a\\b", "..", "-leading-dash", "x" * 65])
def test_invalid_session_ids(session_id):
    with pytest.raises(ValueError):
        validate_session_id(session_id)


def test_chat_ingestor_rejects_path_traversal(workdir, fake_models):
    with pytest.raises(DocuFlowException):
        ChatIngestor(temp_base="data", faiss_base="faiss_index", session_id="../../outside")
    assert not (workdir.parent / "outside").exists()
