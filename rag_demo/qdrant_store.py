from __future__ import annotations

import os

import qdrant_client
from typing import Optional, Sequence, TypedDict
from urllib.parse import urlparse
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore

from rag_demo.config import AppConfig


class QdrantClientKwargs(TypedDict):
    url: str
    api_key: Optional[str]
    timeout: int
    trust_env: bool


def _build_qdrant_client_kwargs(config: AppConfig) -> QdrantClientKwargs:
    _ensure_qdrant_no_proxy(config)
    return {
        "url": config.qdrant_url,
        "api_key": config.qdrant_api_key,
        "timeout": config.qdrant_timeout,
        "trust_env": config.qdrant_trust_env,
    }


def _ensure_qdrant_no_proxy(config: AppConfig) -> None:
    if config.qdrant_trust_env:
        return

    parsed = urlparse(config.qdrant_url)
    host = parsed.hostname
    if not host:
        return

    for env_name in ("NO_PROXY", "no_proxy"):
        current_value = os.getenv(env_name, "")
        entries = [entry.strip() for entry in current_value.split(",") if entry.strip()]
        if host not in entries:
            os.environ[env_name] = ",".join([*entries, host]) if entries else host


def create_qdrant_client(config: AppConfig) -> qdrant_client.QdrantClient:
    return qdrant_client.QdrantClient(**_build_qdrant_client_kwargs(config))


def create_async_qdrant_client(config: AppConfig) -> qdrant_client.AsyncQdrantClient:
    return qdrant_client.AsyncQdrantClient(**_build_qdrant_client_kwargs(config))


def ensure_qdrant_connection(config: AppConfig) -> None:
    client = create_qdrant_client(config)
    try:
        client.get_collections()
    except Exception as exc:
        details = str(exc)
        message_lines = [
            f"Failed to connect to Qdrant at {config.qdrant_url}.",
            f"Original error: {details}",
        ]
        if "502" in details or "Bad Gateway" in details:
            message_lines.append(
                "The request is likely being routed through a local HTTP proxy. "
                "For private IP Qdrant endpoints, keep QDRANT_TRUST_ENV=false and add the Qdrant host to NO_PROXY/no_proxy."
            )
        raise RuntimeError(" ".join(message_lines)) from exc
    finally:
        client.close()


def delete_ref_docs(config: AppConfig, ref_doc_ids: Sequence[str]) -> int:
    unique_ref_doc_ids = list(dict.fromkeys(ref_doc_ids))
    if not unique_ref_doc_ids:
        return 0

    client = create_qdrant_client(config)
    try:
        if not client.collection_exists(config.collection_name):
            return 0

        vector_store = QdrantVectorStore(
            client=client,
            collection_name=config.collection_name,
        )
        for ref_doc_id in unique_ref_doc_ids:
            vector_store.delete(ref_doc_id)
    finally:
        client.close()

    return len(unique_ref_doc_ids)


def create_vector_store(config: AppConfig) -> QdrantVectorStore:
    client = create_qdrant_client(config)
    aclient = create_async_qdrant_client(config)
    return QdrantVectorStore(
        client=client,
        aclient=aclient,
        collection_name=config.collection_name,
    )


def create_storage_context(config: AppConfig) -> StorageContext:
    return StorageContext.from_defaults(vector_store=create_vector_store(config))


def load_index_from_qdrant(config: AppConfig) -> VectorStoreIndex:
    vector_store = create_vector_store(config)
    return VectorStoreIndex.from_vector_store(vector_store=vector_store)
