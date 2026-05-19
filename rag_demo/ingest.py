from __future__ import annotations

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

from rag_demo.config import DOCS_DIR, configure_llamaindex, load_config
from rag_demo.hybrid_splitter import HybridTextSplitter
from rag_demo.markdown_images import augment_markdown_documents_with_images
from rag_demo.qdrant_store import create_storage_context


SUPPORTED_EXTS = [".txt", ".md", ".pdf"]


def ingest() -> None:
    config = load_config()
    configure_llamaindex(config)

    documents = SimpleDirectoryReader(
        input_dir=str(DOCS_DIR),
        recursive=True,
        required_exts=SUPPORTED_EXTS,
    ).load_data()

    if not documents:
        raise RuntimeError(f"No supported documents found in {DOCS_DIR}")

    documents = augment_markdown_documents_with_images(documents, config)

    splitter = HybridTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    nodes = splitter.get_nodes_from_documents(documents, show_progress=True)

    storage_context = create_storage_context(config)
    VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        show_progress=True,
    )

    print(f"Indexed {len(documents)} document(s), {len(nodes)} chunk(s).")
    print(f"Collection: {config.collection_name}")
