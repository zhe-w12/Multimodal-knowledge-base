from types import SimpleNamespace

from manufacturing_rag.adapters.embedding import HashEmbeddingAdapter
from manufacturing_rag.adapters.vector_store import MilvusVectorStore
from manufacturing_rag.config import RAGConfig
from manufacturing_rag.adapters.vector_store import hybrid_candidate_limit
from manufacturing_rag.schemas import EntityExtractionResult


def test_milvus_status_without_env_is_safe():
    config = RAGConfig(zilliz_endpoint_env="MISSING_ZILLIZ_ENDPOINT", zilliz_token_env="MISSING_ZILLIZ_TOKEN")
    status = MilvusVectorStore(config=config).status().to_dict()

    assert status["provider"] == "milvus"
    assert status["connected"] is False
    assert status["endpoint_configured"] is False
    assert status["token_configured"] is False
    assert "token" in status["message"].lower()


def test_hybrid_candidate_limit_overfetches_and_caps():
    assert hybrid_candidate_limit(1) == 20
    assert hybrid_candidate_limit(5) == 20
    assert hybrid_candidate_limit(30) == 100


def test_delete_chunk_ids_flushes_once_after_all_batches(monkeypatch):
    events = []
    progress = []
    collection = SimpleNamespace(
        delete=lambda expression: events.append(("delete", expression)),
        flush=lambda: events.append(("flush", None)),
    )
    utility = SimpleNamespace(has_collection=lambda *_args, **_kwargs: True)
    config = RAGConfig(
        vector_store_provider="milvus",
        embedding_provider="hash",
        embedding_dimensions=64,
    )
    store = MilvusVectorStore(config=config, embedding=HashEmbeddingAdapter(config))
    monkeypatch.setattr(store, "_connect_and_import", lambda: (utility, lambda *_args, **_kwargs: collection))

    result = store.delete_chunk_ids(
        [f"chunk-{index}" for index in range(123)],
        progress_callback=lambda done, total: progress.append((done, total)),
        batch_size=50,
    )

    assert result["deleted"] == 123
    assert [kind for kind, _ in events] == ["delete", "delete", "delete", "flush"]
    assert progress == [(50, 123), (100, 123), (123, 123)]


def test_authenticated_vector_search_filters_before_dense_and_sparse_recall(monkeypatch):
    config = RAGConfig(
        vector_store_provider="milvus",
        embedding_provider="hash",
        embedding_dimensions=64,
        zilliz_endpoint_env="TEST_ENDPOINT",
        zilliz_token_env="TEST_TOKEN",
    )
    collection = SimpleNamespace(
        schema=SimpleNamespace(fields=[SimpleNamespace(name="workspace_id")]),
        load=lambda: None,
    )
    calls = []

    def search(**kwargs):
        calls.append(kwargs)
        return [[]]

    collection.search = search
    store = MilvusVectorStore(config=config, embedding=HashEmbeddingAdapter(config))
    monkeypatch.setattr(store, "_connect_and_import", lambda: (object(), lambda *_args, **_kwargs: collection))
    monkeypatch.setattr(store, "ensure_collection", lambda **_kwargs: None)

    results = store.search(
        "奖学金制度",
        EntityExtractionResult(raw_query="奖学金制度"),
        workspace_ids=["workspace-b", "campus-default", "workspace-a"],
    )

    assert results == []
    assert calls
    expected = 'workspace_id in ["campus-default", "workspace-a", "workspace-b"]'
    assert all(call["expr"] == expected for call in calls)


def test_authenticated_vector_search_fails_closed_on_legacy_collection(monkeypatch):
    config = RAGConfig(
        vector_store_provider="milvus",
        embedding_provider="hash",
        embedding_dimensions=64,
        zilliz_endpoint_env="TEST_ENDPOINT",
        zilliz_token_env="TEST_TOKEN",
    )
    calls = []
    collection = SimpleNamespace(
        schema=SimpleNamespace(fields=[SimpleNamespace(name="metadata_json")]),
        search=lambda **kwargs: calls.append(kwargs) or [[]],
        load=lambda: None,
    )
    store = MilvusVectorStore(config=config, embedding=HashEmbeddingAdapter(config))
    monkeypatch.setattr(store, "_connect_and_import", lambda: (object(), lambda *_args, **_kwargs: collection))
    monkeypatch.setattr(store, "ensure_collection", lambda **_kwargs: None)

    results = store.search(
        "奖学金制度",
        EntityExtractionResult(raw_query="奖学金制度"),
        workspace_ids=["campus-default"],
    )

    assert results == []
    assert calls == []
