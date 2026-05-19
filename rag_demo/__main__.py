from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal RAG demo with LlamaIndex and Qdrant")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("ingest", help="Read docs/ and write embeddings to Qdrant")

    ask_parser = subparsers.add_parser("ask", help="Ask a question from the indexed documents")
    ask_parser.add_argument("question", help="Question to ask")

    args = parser.parse_args()

    if args.command == "ingest":
        from rag_demo.ingest import ingest

        ingest()
    elif args.command == "ask":
        from rag_demo.ask import ask

        ask(args.question)


if __name__ == "__main__":
    main()
