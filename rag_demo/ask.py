from __future__ import annotations

from llama_index.core.prompts import PromptTemplate

from rag_demo.config import AppConfig, configure_llamaindex, load_config
from rag_demo.qdrant_store import load_index_from_qdrant

QA_PROMPT = PromptTemplate(
    "你是一个严谨的 RAG 问答助手。请只根据给定上下文回答问题；"
    "如果上下文没有答案，就直接说不知道。\n\n"
    "上下文：\n{context_str}\n\n"
    "问题：{query_str}\n\n"
    "答案："
)


def ask(question: str) -> None:
    config = load_config()
    configure_llamaindex(config)

    index = load_index_from_qdrant(config)
    query_engine_kwargs = {
        "similarity_top_k": config.similarity_top_k,
        "text_qa_template": QA_PROMPT,
    }
    node_postprocessors = _build_node_postprocessors(config)
    if node_postprocessors:
        query_engine_kwargs["node_postprocessors"] = node_postprocessors
        print(
            "Rerank enabled: "
            f"model={config.rerank_model}, "
            f"retrieve_top_k={config.similarity_top_k}, "
            f"rerank_top_n={config.rerank_top_n}",
            flush=True,
        )

    query_engine = index.as_query_engine(**query_engine_kwargs)
    response = query_engine.query(question)

    print("\n答案：")
    print(str(response).strip())

    print("\n引用来源：")
    for i, source_node in enumerate(response.source_nodes, start=1):
        node = source_node.node
        metadata = node.metadata or {}
        file_name = metadata.get("file_name") or metadata.get("filename") or "unknown"
        file_path = metadata.get("file_path") or metadata.get("path") or ""
        page_label = metadata.get("page_label") or metadata.get("page_number")
        score = source_node.score

        location = file_name
        if page_label is not None:
            location = f"{location}#page={page_label}"

        print(f"[{i}] {location} score={score:.4f}" if score is not None else f"[{i}] {location}")
        if file_path:
            print(f"    path: {file_path}")
        print(f"    text: {_preview(node.get_content())}")


def _build_node_postprocessors(config: AppConfig) -> list[object]:
    if not config.rerank_enabled:
        return []

    print(
        "Loading reranker: "
        f"model={config.rerank_model}, "
        f"retrieve_top_k={config.similarity_top_k}, "
        f"rerank_top_n={config.rerank_top_n}",
        flush=True,
    )
    try:
        from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
    except ImportError as exc:
        raise RuntimeError(
            "RERANK_ENABLED=true, but the BGE reranker dependencies are not installed. "
            "Run `uv sync` or `pip install -r requirements.txt`, then retry."
        ) from exc

    return [
        FlagEmbeddingReranker(
            top_n=config.rerank_top_n,
            model=config.rerank_model,
            use_fp16=config.rerank_use_fp16,
        )
    ]


def _preview(text: str, max_length: int = 180) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_length:
        return normalized
    return normalized[: max_length - 3] + "..."
