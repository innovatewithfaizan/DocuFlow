import pytest

from tests.helpers import HashingEmbeddings, RecordingChatModel
from utils.model_loader import ModelLoader


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """Run in a temp dir so data/, faiss_index/ and logs/ never land in the repo."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def fake_llm() -> RecordingChatModel:
    return RecordingChatModel()


@pytest.fixture
def fake_models(monkeypatch, fake_llm):
    """Route every ModelLoader in the app to the offline doubles in tests/helpers.py."""
    monkeypatch.setattr(ModelLoader, "load_embeddings", lambda self: HashingEmbeddings())
    monkeypatch.setattr(ModelLoader, "load_llm", lambda self, provider_key=None: fake_llm)
    return fake_llm
