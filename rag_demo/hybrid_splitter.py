from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

from llama_index.core.schema import BaseNode, Document, TextNode


_MARKDOWN_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_NUMBERED_HEADING_RE = re.compile(r"^(\d+(?:\.\d+){0,5})[\.、\s]+(.+?)\s*$")
_CHINESE_HEADING_RE = re.compile(r"^([一二三四五六七八九十百]+)[、.．]\s*(.+?)\s*$")
_PAREN_CHINESE_HEADING_RE = re.compile(r"^[（(]([一二三四五六七八九十百]+)[）)]\s*(.+?)\s*$")
_PAREN_NUMBER_HEADING_RE = re.compile(r"^[（(](\d+)[）)]\s*(.+?)\s*$")

_RECURSIVE_SEPARATORS = (
    "\n\n",
    "\n",
    "。", "！", "？", "；",
    ". ", "! ", "? ", "; ",
    "，", ", ",
    " ",
)


@dataclass(frozen=True)
class StructuredBlock:
    text: str
    section_path: Sequence[str]
    heading_level: int


class HybridTextSplitter:
    """Structured section splitting followed by recursive body splitting."""

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be greater than or equal to 0")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def get_nodes_from_documents(
        self,
        documents: Iterable[Document],
        show_progress: bool = False,
    ) -> list[BaseNode]:
        nodes: list[BaseNode] = []

        for document_index, document in enumerate(documents):
            blocks = split_structured(document.get_content())
            for block_index, block in enumerate(blocks):
                chunks = recursive_split(block.text, self.chunk_size, self.chunk_overlap)
                for chunk_index, chunk in enumerate(chunks):
                    metadata = dict(document.metadata or {})
                    metadata.update(
                        {
                            "split_strategy": (
                                "structured"
                                if len(chunks) == 1
                                else "structured_recursive"
                            ),
                            "section_path": " > ".join(block.section_path),
                            "heading_level": block.heading_level,
                            "structure_index": block_index,
                            "chunk_index": chunk_index,
                        }
                    )
                    nodes.append(TextNode(text=chunk, metadata=metadata))

            if show_progress:
                print(
                    f"Split document {document_index + 1}: "
                    f"{len(blocks)} section(s), {len(nodes)} total chunk(s)."
                )

        return nodes


def split_structured(text: str) -> list[StructuredBlock]:
    lines = text.splitlines()
    heading_stack: dict[int, str] = {}
    blocks: list[StructuredBlock] = []
    current_lines: list[str] = []
    current_path: list[str] = []
    current_level = 0

    def flush() -> None:
        nonlocal current_lines, current_path, current_level

        block_text = "\n".join(current_lines).strip()
        if block_text:
            blocks.append(
                StructuredBlock(
                    text=block_text,
                    section_path=tuple(current_path),
                    heading_level=current_level,
                )
            )
        current_lines = []

    for line in lines:
        heading = _parse_heading(line)
        if heading is None:
            current_lines.append(line)
            continue

        flush()

        level, title = heading
        heading_stack = {k: v for k, v in heading_stack.items() if k < level}
        heading_stack[level] = title
        current_path = [heading_stack[k] for k in sorted(heading_stack)]
        current_level = level
        current_lines.append(line)

    flush()

    if blocks:
        return blocks

    stripped = text.strip()
    if not stripped:
        return []
    return [StructuredBlock(text=stripped, section_path=(), heading_level=0)]


def recursive_split(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    chunks = _split_recursive(text.strip(), chunk_size, _RECURSIVE_SEPARATORS)
    return _merge_with_overlap(chunks, chunk_size, chunk_overlap)


def _split_recursive(text: str, chunk_size: int, separators: Sequence[str]) -> list[str]:
    if len(text) <= chunk_size:
        return [text] if text else []

    if not separators:
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    separator = separators[0]
    pieces = _split_keep_separator(text, separator)

    if len(pieces) == 1:
        return _split_recursive(text, chunk_size, separators[1:])

    chunks: list[str] = []
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        if len(piece) <= chunk_size:
            chunks.append(piece)
        else:
            chunks.extend(_split_recursive(piece, chunk_size, separators[1:]))
    return chunks


def _merge_with_overlap(
    pieces: Sequence[str],
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    chunks: list[str] = []
    current = ""

    for piece in pieces:
        if not current:
            current = piece
            continue

        candidate = f"{current}\n{piece}"
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        chunks.append(current)
        overlap = _tail_text(current, chunk_overlap)
        current = f"{overlap}\n{piece}" if overlap else piece

        while len(current) > chunk_size:
            chunks.append(current[:chunk_size])
            overlap = _tail_text(chunks[-1], chunk_overlap)
            current = f"{overlap}{current[chunk_size:]}" if overlap else current[chunk_size:]

    if current:
        chunks.append(current)

    return chunks


def _split_keep_separator(text: str, separator: str) -> list[str]:
    if separator not in text:
        return [text]

    pieces = text.split(separator)
    result: list[str] = []
    for index, piece in enumerate(pieces):
        if not piece:
            continue
        if index < len(pieces) - 1 and separator.strip():
            result.append(piece + separator.rstrip())
        else:
            result.append(piece)
    return result


def _tail_text(text: str, max_length: int) -> str:
    if max_length <= 0:
        return ""
    if len(text) <= max_length:
        return text
    return text[-max_length:]


def _parse_heading(line: str) -> tuple[int, str] | None:
    stripped = line.strip()
    if not stripped:
        return None

    markdown_match = _MARKDOWN_HEADING_RE.match(stripped)
    if markdown_match:
        return len(markdown_match.group(1)), markdown_match.group(2).strip()

    numbered_match = _NUMBERED_HEADING_RE.match(stripped)
    if numbered_match:
        return numbered_match.group(1).count(".") + 1, numbered_match.group(2).strip()

    chinese_match = _CHINESE_HEADING_RE.match(stripped)
    if chinese_match:
        return 1, chinese_match.group(2).strip()

    paren_chinese_match = _PAREN_CHINESE_HEADING_RE.match(stripped)
    if paren_chinese_match:
        return 2, paren_chinese_match.group(2).strip()

    paren_number_match = _PAREN_NUMBER_HEADING_RE.match(stripped)
    if paren_number_match:
        return 3, paren_number_match.group(2).strip()

    return None
