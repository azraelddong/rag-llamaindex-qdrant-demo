from __future__ import annotations

import base64
import mimetypes
import re
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import unquote, urlparse

from llama_index.core import Document
from openai import OpenAI

from rag_demo.config import AppConfig, DOCS_DIR


_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
_SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
_IMAGE_PROMPT = (
    "请解析这张 Markdown 文档中的图片内容，用中文输出适合 RAG 检索入库的文字描述。"
    "如果图片包含表格、流程图、截图、代码、公式或文字，请尽量完整提取关键文字、"
    "结构关系、字段名称、数值和结论。不要编造图片中不存在的信息。"
)


ImageDescriber = Callable[[Path, str], str]


def augment_markdown_documents_with_images(
    documents: Iterable[Document],
    config: AppConfig,
) -> list[Document]:
    if not config.image_parse_enabled:
        return list(documents)

    describer = OpenAIVisionImageDescriber(config)
    augmented: list[Document] = []
    total_images = 0

    for document in documents:
        file_path = _document_path(document)
        if file_path is None or file_path.suffix.lower() not in {".md", ".markdown"}:
            augmented.append(document)
            continue

        content, image_count = augment_markdown_image_content(
            text=document.get_content(),
            markdown_path=file_path,
            describe_image=describer.describe,
        )
        total_images += image_count
        augmented.append(Document(text=content, metadata=dict(document.metadata or {})))

    if total_images:
        print(f"Parsed {total_images} Markdown image(s) into text for indexing.")

    return augmented


def augment_markdown_image_content(
    text: str,
    markdown_path: Path,
    describe_image: ImageDescriber,
) -> tuple[str, int]:
    image_count = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal image_count

        alt_text = match.group(1).strip()
        image_target = _parse_image_target(match.group(2))
        image_path = _resolve_image_path(markdown_path, image_target)

        print(f"Processing Markdown image: target={image_target}, alt_text={alt_text}, resolved_path={image_path}")

        if image_path is None:
            print(f"Skip remote Markdown image: {image_target}")
            return match.group(0)
        if not image_path.exists():
            print(f"Skip missing Markdown image: {image_path}")
            return match.group(0)
        if image_path.suffix.lower() not in _SUPPORTED_IMAGE_EXTS:
            print(f"Skip unsupported Markdown image type: {image_path}")
            return match.group(0)

        description = describe_image(image_path, alt_text)
        image_count += 1
        return f"{match.group(0)}\n\n{_image_description_block(image_target, alt_text, description)}"

    return _IMAGE_RE.sub(replace, text), image_count


class OpenAIVisionImageDescriber:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.client = OpenAI(
            api_key=config.image_api_key,
            base_url=config.image_api_base,
            timeout=config.ai_timeout,
        )

    def describe(self, image_path: Path, alt_text: str) -> str:
        image_url = _image_data_url(image_path)
        prompt = _IMAGE_PROMPT
        if alt_text:
            prompt = f"{prompt}\n\nMarkdown 图片替代文本：{alt_text}"

        response = self.client.chat.completions.create(
            model=self.config.image_model,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url,
                                "detail": self.config.image_detail,
                            },
                        },
                    ],
                }
            ],
        )
        print(f"解析markdown图片返回的结果 : {response}")
        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError(f"Vision model returned empty content for image: {image_path}")
        return content.strip()


def _document_path(document: Document) -> Path | None:
    metadata = document.metadata or {}
    path = metadata.get("file_path") or metadata.get("path")
    if path is None:
        return None
    return Path(str(path))


def _parse_image_target(raw_target: str) -> str:
    target = raw_target.strip()
    if target.startswith("<") and ">" in target:
        return target[1 : target.index(">")].strip()

    for quote in ('"', "'"):
        marker = f" {quote}"
        if marker in target:
            target = target[: target.index(marker)]
            break

    return target.strip()


def _resolve_image_path(markdown_path: Path, image_target: str) -> Path | None:
    parsed = urlparse(image_target)
    if parsed.scheme in {"http", "https", "data"}:
        return None

    target_path = Path(unquote(image_target))
    if target_path.is_absolute():
        return target_path

    markdown_dir = markdown_path.parent if markdown_path.parent else DOCS_DIR
    return (markdown_dir / target_path).resolve()


def _image_description_block(image_target: str, alt_text: str, description: str) -> str:
    lines = [
        "[图片解析]",
        f"图片路径: {image_target}",
    ]
    if alt_text:
        lines.append(f"图片替代文本: {alt_text}")
    lines.extend(
        [
            f"图片内容: {description}",
            "[/图片解析]",
        ]
    )
    return "\n".join(lines)


def _image_data_url(image_path: Path) -> str:
    mime_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"
