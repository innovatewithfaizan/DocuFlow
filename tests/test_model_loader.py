import json
import os

from utils.model_loader import _load_bundled_api_keys


def test_bundled_api_keys_are_expanded(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("API_KEYS", json.dumps({"GROQ_API_KEY": "g-test", "OPENAI_API_KEY": "o-test"}))

    _load_bundled_api_keys()

    assert os.environ["GROQ_API_KEY"] == "g-test"
    assert os.environ["OPENAI_API_KEY"] == "o-test"


def test_explicit_env_vars_win_over_bundle(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "explicit")
    monkeypatch.setenv("API_KEYS", json.dumps({"OPENAI_API_KEY": "from-bundle"}))

    _load_bundled_api_keys()

    assert os.environ["OPENAI_API_KEY"] == "explicit"


def test_invalid_bundle_is_ignored(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("API_KEYS", "not-json")

    _load_bundled_api_keys()

    assert "GROQ_API_KEY" not in os.environ
