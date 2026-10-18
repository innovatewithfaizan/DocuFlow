import os
import sys
import json
from typing import Optional
from dotenv import load_dotenv
from utils.config_loader import load_config
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocuFlowException
log = CustomLogger().get_logger(__name__)

# The API key each provider needs. "fastembed" runs locally and needs none.
PROVIDER_KEYS = {
    "openai": "OPENAI_API_KEY",
    "groq": "GROQ_API_KEY",
    "google": "GOOGLE_API_KEY",
    "fastembed": None,
}


def _load_bundled_api_keys() -> None:
    """
    On ECS the task definition injects every key as one JSON secret named API_KEYS
    (see .github/workflows/task_definition.json). Expand it into the individual
    variables the loaders read. Variables that are already set take precedence.
    """
    raw = os.getenv("API_KEYS")
    if not raw:
        return
    try:
        bundle = json.loads(raw)
    except json.JSONDecodeError:
        log.warning("API_KEYS is set but is not valid JSON; ignoring it")
        return
    if not isinstance(bundle, dict):
        log.warning("API_KEYS must be a JSON object; ignoring it")
        return
    for key, value in bundle.items():
        if value and not os.getenv(key):
            os.environ[key] = str(value)


class ModelLoader:
    """
    Loads the embedding model and the LLM selected in config/config.yaml.

    EMBEDDING_PROVIDER and LLM_PROVIDER override the config per process. Only the
    API keys of the providers actually used are required, and each is checked when
    its model is loaded rather than up front.
    """

    def __init__(self):
        load_dotenv()
        _load_bundled_api_keys()
        self.config = load_config()
        self.embedding_provider = os.getenv(
            "EMBEDDING_PROVIDER", self.config["embedding_model"].get("provider", "openai")
        )
        self.llm_provider = os.getenv("LLM_PROVIDER", self.config.get("default_llm", "openai"))
        log.info("Configuration loaded successfully", config_keys=list(self.config.keys()))

    @staticmethod
    def _require_key(provider: str) -> Optional[str]:
        if provider not in PROVIDER_KEYS:
            raise ValueError(f"Unsupported provider: {provider}")
        env_var = PROVIDER_KEYS[provider]
        if env_var is None:
            return None
        value = os.getenv(env_var)
        if not value:
            log.error("Missing environment variable", missing_var=env_var, provider=provider)
            raise DocuFlowException(f"Missing environment variable {env_var} for provider '{provider}'", sys)
        return value

    def load_embeddings(self):
        """
        Load and return the embedding model.
        """
        emb_cfg = self.config["embedding_model"]
        provider = self.embedding_provider
        api_key = self._require_key(provider)
        try:
            log.info("Loading embedding model...", provider=provider)
            if provider == "openai":
                from langchain_openai import OpenAIEmbeddings
                return OpenAIEmbeddings(model=emb_cfg["model_name"], api_key=api_key)  # type: ignore
            if provider == "google":
                from langchain_google_genai import GoogleGenerativeAIEmbeddings
                return GoogleGenerativeAIEmbeddings(model=emb_cfg["google_model_name"], google_api_key=api_key)  # type: ignore
            if provider == "fastembed":
                # Local ONNX model, no key or network after the first download.
                # Not in requirements.txt: pip install -r evaluation/requirements.txt
                from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
                return FastEmbedEmbeddings(model_name=emb_cfg["local_model_name"])
            raise ValueError(f"Unsupported embedding provider: {provider}")
        except Exception as e:
            log.error("Error loading embedding model", error=str(e))
            raise DocuFlowException("Failed to load embedding model", e) from e

    def load_llm(self, provider_key: Optional[str] = None, model_name: Optional[str] = None):
        """Load the LLM for `provider_key` (defaults to LLM_PROVIDER, then config default_llm).
        `model_name` overrides the configured model, e.g. to use a different judge model."""
        llm_block = self.config["llm"]
        provider_key = provider_key or self.llm_provider

        log.info("Loading LLM...")
        if provider_key not in llm_block:
            log.error("LLM provider not found in config", provider_key=provider_key)
            raise ValueError(f"Provider '{provider_key}' not found in config")

        llm_config = llm_block[provider_key]
        provider = llm_config.get("provider")
        model_name = model_name or llm_config.get("model_name")
        temperature = llm_config.get("temperature", 0)
        max_tokens = llm_config.get("max_output_tokens", 2048)
        api_key = self._require_key(provider)

        log.info("Loading LLM", provider=provider, model=model_name, temperature=temperature, max_tokens=max_tokens)

        if provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,  # type: ignore
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
        if provider == "groq":
            from langchain_groq import ChatGroq
            return ChatGroq(
                model=model_name,
                api_key=api_key,  # type: ignore
                temperature=temperature,
                max_tokens=max_tokens,
            )
        if provider == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=model_name,
                api_key=api_key,  # type: ignore
                temperature=temperature,
                max_tokens=max_tokens,
                max_retries=6,
            )
        log.error("Unsupported LLM provider", provider=provider)
        raise ValueError(f"Unsupported LLM provider: {provider}")


if __name__ == "__main__":
    loader = ModelLoader()

    embeddings = loader.load_embeddings()
    print(f"Embedding Model Loaded: {embeddings}")
    result = embeddings.embed_query("Hello, how are you?")
    print(f"Embedding dimension: {len(result)}")

    llm = loader.load_llm()
    print(f"LLM Loaded: {llm}")
    result = llm.invoke("Hello, how are you?")
    print(f"LLM Result: {result.content}")
