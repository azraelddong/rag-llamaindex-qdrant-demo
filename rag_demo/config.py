from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, cast

from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "docs"

ImageDetail = Literal["auto", "low", "high"]


@dataclass(frozen=True)
class AppConfig:
    collection_name: str
    qdrant_url: str
    qdrant_api_key: Optional[str]
    qdrant_timeout: int
    qdrant_trust_env: bool
    chunk_size: int
    chunk_overlap: int
    similarity_top_k: int
    rerank_enabled: bool
    rerank_model: str
    rerank_top_n: int
    rerank_use_fp16: bool
    api_base: str
    api_key: str
    chat_api_base: str
    chat_api_key: str
    embed_api_base: str
    embed_api_key: str
    chat_model: str
    embed_model: str
    embed_dimensions: Optional[int]
    image_parse_enabled: bool
    image_api_base: str
    image_api_key: str
    image_model: str
    image_detail: ImageDetail
    ai_timeout: float


def load_config() -> AppConfig:
    load_dotenv(PROJECT_ROOT / ".env")
    api_base = _require_env("AI_API_BASE")
    api_key = _require_env("AI_API_KEY")

    chat_api_base = _blank_to_none(os.getenv("CHAT_API_BASE")) or api_base
    chat_api_key = _blank_to_none(os.getenv("CHAT_API_KEY")) or api_key
    chat_model = _require_env("CHAT_MODEL")

    return AppConfig(
        collection_name=os.getenv("QDRANT_COLLECTION", "rag_demo"),
        qdrant_url=_require_env("QDRANT_URL"),
        qdrant_api_key=_blank_to_none(os.getenv("QDRANT_API_KEY")),
        qdrant_timeout=int(os.getenv("QDRANT_TIMEOUT", "60")),
        qdrant_trust_env=_optional_bool(os.getenv("QDRANT_TRUST_ENV"), default=False),
        chunk_size=int(os.getenv("CHUNK_SIZE", "512")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "80")),
        similarity_top_k=int(os.getenv("SIMILARITY_TOP_K", "20")),
        rerank_enabled=_optional_bool(os.getenv("RERANK_ENABLED"), default=True),
        rerank_model=_blank_to_none(os.getenv("RERANK_MODEL")) or "BAAI/bge-reranker-v2-m3",
        rerank_top_n=int(os.getenv("RERANK_TOP_N", "4")),
        rerank_use_fp16=_optional_bool(os.getenv("RERANK_USE_FP16"), default=False),
        api_base=api_base,
        api_key=api_key,
        chat_api_base=chat_api_base,
        chat_api_key=chat_api_key,
        embed_api_base=_blank_to_none(os.getenv("EMBED_API_BASE")) or api_base,
        embed_api_key=_blank_to_none(os.getenv("EMBED_API_KEY")) or api_key,
        chat_model=chat_model,
        embed_model=_require_env("EMBED_MODEL"),
        embed_dimensions=_optional_int(os.getenv("EMBED_DIMENSIONS")),
        image_parse_enabled=_optional_bool(os.getenv("IMAGE_PARSE_ENABLED"), default=True),
        image_api_base=_blank_to_none(os.getenv("IMAGE_API_BASE")) or chat_api_base,
        image_api_key=_blank_to_none(os.getenv("IMAGE_API_KEY")) or chat_api_key,
        image_model=_blank_to_none(os.getenv("IMAGE_MODEL")) or chat_model,
        image_detail=_load_image_detail(),
        ai_timeout=float(os.getenv("AI_TIMEOUT", "120")),
    )


def configure_llamaindex(config: AppConfig) -> None:
    Settings.chunk_size = config.chunk_size
    Settings.chunk_overlap = config.chunk_overlap
    embed_kwargs = {
        "model": config.embed_model,
        "api_key": config.embed_api_key,
        "api_base": config.embed_api_base,
        "timeout": config.ai_timeout,
    }
    if config.embed_dimensions is not None:
        embed_kwargs["dimensions"] = config.embed_dimensions
    Settings.embed_model = OpenAIEmbedding(**embed_kwargs)
    Settings.llm = OpenAI(
        model=config.chat_model,
        api_key=config.chat_api_key,
        api_base=config.chat_api_base,
        temperature=0.1,
        timeout=config.ai_timeout,
    )


def _blank_to_none(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _require_env(name: str) -> str:
    value = _blank_to_none(os.getenv(name))
    if value is None:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _optional_int(value: Optional[str]) -> Optional[int]:
    value = _blank_to_none(value)
    if value is None:
        return None
    return int(value)


def _optional_bool(value: Optional[str], default: bool) -> bool:
    value = _blank_to_none(value)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _load_image_detail() -> ImageDetail:
    value = (os.getenv("IMAGE_DETAIL", "auto") or "auto").strip().lower()
    if value in {"auto", "low", "high"}:
        return cast(ImageDetail, value)
    raise RuntimeError("IMAGE_DETAIL must be one of: auto, low, high")
