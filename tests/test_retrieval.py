"""Retrieval and the ConversationalRAG chain, with the LLM mocked."""
import pytest

from exception.custom_exception import GroundworkException
from src.document_chat.retrieval import ConversationalRAG
from src.document_ingestion.data_ingestion import ChatIngestor
from tests.helpers import FakeUpload

TOPICS = {
    "volcano": "Mount Etna is an active volcano. Lava flows and ash eruptions are monitored by volcanologists.",
    "tea": "Darjeeling tea grows in the Himalayan foothills. The first flush harvest happens in spring.",
    "chess": "The Najdorf is a chess opening line. Grandmasters study the opening with engines.",
}


@pytest.fixture
def indexed_session(workdir, fake_models):
    paths = []
    for name, text in TOPICS.items():
        p = workdir / f"{name}.txt"
        p.write_text(text, encoding="utf-8")
        paths.append(FakeUpload(p))
    ingestor = ChatIngestor(temp_base="data", faiss_base="faiss_index", session_id="rag")
    retriever = ingestor.build_retriever(paths, chunk_size=500, chunk_overlap=0, k=1)
    return ingestor, retriever


def test_retriever_returns_the_relevant_chunk(indexed_session):
    _, retriever = indexed_session

    docs = retriever.invoke("When is the first flush tea harvest?")

    assert len(docs) == 1
    assert "Darjeeling" in docs[0].page_content


def test_rag_answers_from_retrieved_context(indexed_session, fake_llm):
    ingestor, _ = indexed_session
    fake_llm.answer = "Lava and ash are monitored."
    rag = ConversationalRAG(session_id="rag")
    rag.load_retriever_from_faiss(str(ingestor.faiss_dir), k=1)

    answer = rag.invoke("What do volcanologists monitor on Etna?", chat_history=[])

    assert answer == "Lava and ash are monitored."
    [context] = fake_llm.qa_contexts()
    assert "Mount Etna" in context
    assert "Darjeeling" not in context and "Najdorf" not in context


def test_k_controls_how_many_chunks_reach_the_llm(indexed_session, fake_llm):
    ingestor, _ = indexed_session
    rag = ConversationalRAG(session_id="rag")
    rag.load_retriever_from_faiss(str(ingestor.faiss_dir), k=3)

    rag.invoke("Tell me about Etna, tea and chess openings", chat_history=[])

    [context] = fake_llm.qa_contexts()
    assert all(marker in context for marker in ("Mount Etna", "Darjeeling", "Najdorf"))


def test_invoke_before_loading_a_retriever_fails_clearly(fake_models):
    rag = ConversationalRAG(session_id="empty")
    with pytest.raises(GroundworkException):
        rag.invoke("anything")


def test_missing_index_directory_fails_clearly(workdir, fake_models):
    rag = ConversationalRAG(session_id="missing")
    with pytest.raises(GroundworkException):
        rag.load_retriever_from_faiss(str(workdir / "does-not-exist"))


def test_answers_cite_their_sources(indexed_session, fake_llm):
    ingestor, _ = indexed_session
    rag = ConversationalRAG(session_id="rag")
    rag.load_retriever_from_faiss(str(ingestor.faiss_dir), k=1, search_type="similarity")

    result = rag.invoke_with_sources("When is the first flush tea harvest?")

    assert result["sources"] == [{"source": "tea.txt", "page": None, "snippet": TOPICS["tea"][:200]}]
    assert result["contexts"] == [TOPICS["tea"]]


def test_first_question_skips_the_rewrite_call(indexed_session, fake_llm):
    ingestor, _ = indexed_session
    rag = ConversationalRAG(session_id="rag")
    rag.load_retriever_from_faiss(str(ingestor.faiss_dir), k=1)

    rag.invoke("What is the Najdorf?", chat_history=[])

    assert len(fake_llm.prompts) == 1  # only the answer prompt, no rewrite


def test_follow_up_questions_are_rewritten_before_retrieval(indexed_session, fake_llm):
    from langchain_core.messages import AIMessage, HumanMessage
    ingestor, _ = indexed_session
    rag = ConversationalRAG(session_id="rag")
    rag.load_retriever_from_faiss(str(ingestor.faiss_dir), k=1)
    history = [HumanMessage("Tell me about Darjeeling tea"), AIMessage("It grows in the Himalayan foothills.")]

    result = rag.invoke_with_sources("When is the first flush harvest?", chat_history=history)

    assert len(fake_llm.prompts) == 2  # rewrite + answer
    assert result["sources"][0]["source"] == "tea.txt"
