"""HTTP-level tests for the chat endpoints, with embeddings and LLM mocked."""
import pytest
from fastapi.testclient import TestClient

from api.main import app

DOCS = {
    "etna.txt": "Mount Etna is an active volcano. Lava flows and ash eruptions are monitored by volcanologists.",
    "tea.txt": "Darjeeling tea grows in the Himalayan foothills. The first flush harvest happens in spring.",
    "chess.txt": "The Najdorf is a chess opening line. Grandmasters study the opening with engines.",
}


@pytest.fixture
def client(workdir, fake_models):
    return TestClient(app)


def build_index(client, **form):
    files = [("files", (name, text.encode(), "text/plain")) for name, text in DOCS.items()]
    resp = client.post("/chat/index", files=files, data={"chunk_size": "500", "chunk_overlap": "0", **form})
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "service": "groundwork"}


def test_index_reports_what_was_indexed(client):
    body = build_index(client)

    assert body["session_id"].startswith("session_")
    assert body["documents"] == 3
    assert body["chunks"] == body["added"] == 3


def test_index_then_query_round_trip(client, fake_llm):
    fake_llm.answer = "In spring."
    session_id = build_index(client)["session_id"]

    resp = client.post("/chat/query", data={"question": "When is the first flush harvest?",
                                            "session_id": session_id, "k": "1"})

    assert resp.status_code == 200, resp.text
    assert resp.json()["answer"] == "In spring."
    [context] = fake_llm.qa_contexts()
    assert "Darjeeling" in context and "Etna" not in context


def test_query_honours_k(client, fake_llm):
    # Regression: /chat/query accepted k but always retrieved 5 chunks
    session_id = build_index(client)["session_id"]

    client.post("/chat/query", data={"question": "volcano tea chess", "session_id": session_id, "k": "2"})

    [context] = fake_llm.qa_contexts()
    assert sum(marker in context for marker in ("Etna", "Darjeeling", "Najdorf")) == 2


def test_query_unknown_session_is_404(client):
    resp = client.post("/chat/query", data={"question": "hi", "session_id": "session_does_not_exist"})
    assert resp.status_code == 404


@pytest.mark.parametrize("endpoint", ["/chat/query", "/chat/index"])
def test_path_traversal_session_ids_are_rejected(client, endpoint):
    data = {"question": "hi", "session_id": "../../etc"}
    files = [("files", ("a.txt", b"text", "text/plain"))] if endpoint == "/chat/index" else None
    resp = client.post(endpoint, data=data, files=files)
    assert resp.status_code == 400


def test_query_requires_session_id(client):
    resp = client.post("/chat/query", data={"question": "hi"})
    assert resp.status_code == 400


def test_query_returns_sources_and_retrieval_settings(client):
    session_id = build_index(client)["session_id"]

    body = client.post("/chat/query", data={"question": "When is the first flush harvest?",
                                            "session_id": session_id, "k": "1",
                                            "search_type": "similarity"}).json()

    assert body["search_type"] == "similarity"
    assert body["compressed"] is False
    assert [s["source"] for s in body["sources"]] == ["tea.txt"]


def test_query_uses_the_configured_search_type_by_default(client):
    session_id = build_index(client)["session_id"]

    body = client.post("/chat/query", data={"question": "tea", "session_id": session_id}).json()

    assert body["search_type"] == "similarity"


def test_query_rejects_unknown_search_type(client):
    session_id = build_index(client)["session_id"]

    resp = client.post("/chat/query", data={"question": "tea", "session_id": session_id, "search_type": "bm25"})

    assert resp.status_code == 400
