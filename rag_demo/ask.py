from __future__ import annotations

from llama_index.core.prompts import PromptTemplate

from rag_demo.config import configure_llamaindex, load_config
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
    query_engine = index.as_query_engine(
        similarity_top_k=config.similarity_top_k,
        text_qa_template=QA_PROMPT,
    )
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


def _preview(text: str, max_length: int = 180) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_length:
        return normalized
    return normalized[: max_length - 3] + "..."
