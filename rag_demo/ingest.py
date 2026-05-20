from __future__ import annotations

from pathlib import Path

from llama_index.core import Document, SimpleDirectoryReader, VectorStoreIndex

from rag_demo.config import DOCS_DIR, configure_llamaindex, load_config
from rag_demo.hybrid_splitter import HybridTextSplitter
from rag_demo.markdown_images import augment_markdown_documents_with_images
from rag_demo.qdrant_store import (
    create_storage_context,
    delete_ref_docs,
    ensure_qdrant_connection,
)

"""支持从 DOCS_DIR 目录下的文本、Markdown 和 PDF 文件中读取文档，解析 Markdown 图片内容，进行分块，并将结果索引到 Qdrant 中。"""
SUPPORTED_EXTS = [".txt", ".md", ".pdf"]

def _document_ref_doc_id(document: Document) -> str:
    metadata = document.metadata or {}
    file_path = metadata.get("file_path") or metadata.get("path")
    if file_path:
        return str(Path(str(file_path)).resolve())

    file_name = metadata.get("file_name") or metadata.get("filename")
    if file_name:
        return str(file_name)

    return document.doc_id


def _prepare_documents_for_indexing(documents: list[Document]) -> list[str]:
    ref_doc_ids: list[str] = []
    for document in documents:
        ref_doc_id = _document_ref_doc_id(document)
        document.doc_id = ref_doc_id
        ref_doc_ids.append(ref_doc_id)
    return list(dict.fromkeys(ref_doc_ids))


def ingest() -> None:
    config = load_config()
    configure_llamaindex(config)
    ensure_qdrant_connection(config)

    documents = SimpleDirectoryReader(
        input_dir=str(DOCS_DIR),
        recursive=True,
        required_exts=SUPPORTED_EXTS,
    ).load_data()

    if not documents:
        raise RuntimeError(f"No supported documents found in {DOCS_DIR}")

    documents = augment_markdown_documents_with_images(documents, config)
    ref_doc_ids = _prepare_documents_for_indexing(documents)

    splitter = HybridTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    nodes = splitter.get_nodes_from_documents(documents, show_progress=True)

    deleted_count = delete_ref_docs(config, ref_doc_ids)
    storage_context = create_storage_context(config)
    VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        show_progress=True,
    )

    if deleted_count:
        print(f"Replaced {deleted_count} existing document(s) before indexing.")
    print(f"Indexed {len(documents)} document(s), {len(nodes)} chunk(s).")
    print(f"Collection: {config.collection_name}")
