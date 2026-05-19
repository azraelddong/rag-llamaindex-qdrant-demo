# RAG Demo 示例文档

这个项目演示如何使用 Python、LlamaIndex 和 Qdrant 搭建一个最小 RAG 应用。

流程包括：

1. 从 `docs` 目录读取 `txt`、`md`、`pdf` 文件。
2. 解析 Markdown 中的本地图片内容并转成可检索文本。
3. 使用结构化 + 递归式混合策略将文档切分成 chunk。
4. 使用本地 HuggingFace embedding 生成向量。
5. 将向量和来源元数据写入 Qdrant。
6. 在命令行中提问，并返回答案和引用来源。

如果你看到这段内容作为引用，说明索引和检索链路已经跑通。
