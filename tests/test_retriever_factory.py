"""MMR and contextual compression, built by src/document_chat/retriever_factory.py."""
import pytest
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.vectorstores import FAISS

from src.document_chat.retriever_factory import CompressionSettings, RetrieverSettings, build_retriever
from tests.helpers import HashingEmbeddings

# Three near-identical chunks about Etna's lava, and one about a different aspect of Etna.
NEAR_DUPLICATES = [
    "Etna volcano lava flow eruption report from the monitoring station",
    "Etna volcano lava flow eruption report from the monitoring station today",
    "Etna volcano lava flow eruption report from the monitoring station again",
]
DIFFERENT_ASPECT = "Etna volcano region tourism hotels vineyards and wine tasting"
QUERY = "Etna volcano lava flow"


@pytest.fixture
def store():
    return FAISS.from_texts(NEAR_DUPLICATES + [DIFFERENT_ASPECT], HashingEmbeddings())


def test_settings_default_to_config():
    s = RetrieverSettings.from_config()
    assert s.search_type == "similarity"
    assert (s.k, s.fetch_k, s.lambda_mult) == (5, 20, 0.5)
    assert s.compression.enabled is False


def test_settings_overrides_ignore_none():
    s = RetrieverSettings.from_config(k=3, search_type=None, compress=True)
    assert s.k == 3
    assert s.search_type == "similarity"
    assert s.compression.enabled is True


def test_settings_read_a_custom_config_block():
    cfg = {"retriever": {"search_type": "similarity", "top_k": 7, "compression": {"enabled": True, "keep_top_n": 3}}}
    s = RetrieverSettings.from_config(cfg)
    assert (s.search_type, s.k, s.compression.enabled, s.compression.keep_top_n) == ("similarity", 7, True, 3)


def test_similarity_returns_redundant_chunks(store):
    retriever = build_retriever(store, RetrieverSettings(search_type="similarity", k=2))

    texts = [d.page_content for d in retriever.invoke(QUERY)]

    assert all(t in NEAR_DUPLICATES for t in texts)


def test_mmr_trades_a_duplicate_for_new_information(store):
    retriever = build_retriever(store, RetrieverSettings(search_type="mmr", k=2, fetch_k=4, lambda_mult=0.3))

    texts = [d.page_content for d in retriever.invoke(QUERY)]

    assert texts[0] in NEAR_DUPLICATES  # most relevant chunk still comes first
    assert DIFFERENT_ASPECT in texts    # the second slot goes to non-redundant content


def test_mmr_with_lambda_one_behaves_like_similarity(store):
    mmr = build_retriever(store, RetrieverSettings(search_type="mmr", k=2, fetch_k=4, lambda_mult=1.0))
    sim = build_retriever(store, RetrieverSettings(search_type="similarity", k=2))

    assert {d.page_content for d in mmr.invoke(QUERY)} == {d.page_content for d in sim.invoke(QUERY)}


def test_compression_shrinks_context_but_keeps_the_answer():
    filler = " ".join(f"Unrelated sentence {i} about office furniture and printer toner." for i in range(30))
    chunks = [
        f"{filler} The Etna observatory measured a lava temperature of 1100 degrees. {filler}",
        f"{filler} {filler}",
    ]
    store = FAISS.from_texts(chunks, HashingEmbeddings())
    plain = build_retriever(store, RetrieverSettings(search_type="similarity", k=2))
    compressed = build_retriever(
        store,
        RetrieverSettings(search_type="similarity", k=2,
                          compression=CompressionSettings(enabled=True, piece_size=120, keep_top_n=2)),
    )
    question = "What lava temperature did the Etna observatory measure?"

    plain_docs = plain.invoke(question)
    compressed_docs = compressed.invoke(question)

    assert isinstance(compressed, ContextualCompressionRetriever)
    assert sum(len(d.page_content) for d in compressed_docs) < 0.2 * sum(len(d.page_content) for d in plain_docs)
    assert any("1100 degrees" in d.page_content for d in compressed_docs)


def test_compression_drops_near_duplicate_pieces():
    store = FAISS.from_texts(["Lava reached the village of Zafferana.", "Lava reached the village of Zafferana!"],
                             HashingEmbeddings())
    retriever = build_retriever(
        store,
        RetrieverSettings(search_type="similarity", k=2,
                          compression=CompressionSettings(enabled=True, piece_size=200, keep_top_n=5)),
    )

    assert len(retriever.invoke("Which village did the lava reach?")) == 1


@pytest.mark.parametrize("bad", [RetrieverSettings(search_type="bm25"), RetrieverSettings(k=0)])
def test_invalid_settings_are_rejected(store, bad):
    with pytest.raises(ValueError):
        build_retriever(store, bad)
