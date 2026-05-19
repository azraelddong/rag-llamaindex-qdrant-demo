from __future__ import annotations

import qdrant_client
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore

from rag_demo.config import AppConfig


def create_qdrant_client(config: AppConfig) -> qdrant_client.QdrantClient:
    return qdrant_client.QdrantClient(
        url=config.qdrant_url,
        api_key=config.qdrant_api_key,
        timeout=config.qdrant_timeout,
        trust_env=config.qdrant_trust_env,
    )


def create_vector_store(config: AppConfig) -> QdrantVectorStore:
    client = create_qdrant_client(config)
    return QdrantVectorStore(
        client=client,
        collection_name=config.collection_name,
    )


def create_storage_context(config: AppConfig) -> StorageContext:
    return StorageContext.from_defaults(vector_store=create_vector_store(config))


def load_index_from_qdrant(config: AppConfig) -> VectorStoreIndex:
    vector_store = create_vector_store(config)
    return VectorStoreIndex.from_vector_store(vector_store=vector_store)
