import json
import os

import pytest

from exception.custom_exception import DocuFlowException
from utils.model_loader import ModelLoader, _load_bundled_api_keys


@pytest.fixture
def clean_env(monkeypatch):
    for var in ("OPENAI_API_KEY", "GROQ_API_KEY", "GOOGLE_API_KEY", "API_KEYS",
                "LLM_PROVIDER", "EMBEDDING_PROVIDER"):
        monkeypatch.delenv(var, raising=False)
    # keep a developer's local .env out of these tests
    monkeypatch.setattr("utils.model_loader.load_dotenv", lambda *a, **k: None)
    return monkeypatch


def test_bundled_api_keys_are_expanded(clean_env):
    clean_env.setenv("API_KEYS", json.dumps({"GROQ_API_KEY": "g-test", "OPENAI_API_KEY": "o-test"}))

    _load_bundled_api_keys()

    assert os.environ["GROQ_API_KEY"] == "g-test"
    assert os.environ["OPENAI_API_KEY"] == "o-test"


def test_explicit_env_vars_win_over_bundle(clean_env):
    clean_env.setenv("OPENAI_API_KEY", "explicit")
    clean_env.setenv("API_KEYS", json.dumps({"OPENAI_API_KEY": "from-bundle"}))

    _load_bundled_api_keys()

    assert os.environ["OPENAI_API_KEY"] == "explicit"


def test_invalid_bundle_is_ignored(clean_env):
    clean_env.setenv("API_KEYS", "not-json")

    _load_bundled_api_keys()

    assert "GROQ_API_KEY" not in os.environ


def test_only_the_selected_providers_key_is_required(clean_env):
    # Previously all three provider keys were required even to use one of them
    clean_env.setenv("LLM_PROVIDER", "groq")
    clean_env.setenv("GROQ_API_KEY", "g-test")

    llm = ModelLoader().load_llm()

    assert type(llm).__name__ == "ChatGroq"
    assert llm.model_name == "openai/gpt-oss-120b"


def test_missing_key_names_the_variable(clean_env):
    with pytest.raises(DocuFlowException, match="OPENAI_API_KEY"):
        ModelLoader().load_embeddings()


def test_openai_settings_come_from_config(clean_env):
    clean_env.setenv("OPENAI_API_KEY", "o-test")

    loader = ModelLoader()
    llm = loader.load_llm()
    emb = loader.load_embeddings()

    assert llm.model_name == "gpt-4o"
    assert llm.temperature == 0
    # config used max_tokens while the loader read max_output_tokens, so the limit was ignored
    assert llm.max_tokens == 1000
    assert emb.model == "text-embedding-3-small"


def test_unknown_llm_provider_is_rejected(clean_env):
    clean_env.setenv("LLM_PROVIDER", "not-a-provider")

    with pytest.raises(ValueError):
        ModelLoader().load_llm()


def test_github_models_provider_uses_the_openai_compatible_endpoint(clean_env):
    clean_env.setenv("GITHUB_TOKEN", "gh-test")

    loader = ModelLoader()
    llm = loader.load_llm("github")
    judge = loader.load_llm("github", model_name="openai/gpt-4o")

    assert llm.model_name == "openai/gpt-4o-mini"
    assert judge.model_name == "openai/gpt-4o"
    assert str(llm.openai_api_base).startswith("https://models.github.ai")


def test_fastembed_needs_no_api_key(clean_env):
    assert ModelLoader._require_key("fastembed") is None
