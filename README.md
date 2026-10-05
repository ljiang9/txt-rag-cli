# txt-rag-cli

零依赖的纯文本目录 RAG（检索增强生成）命令行工具。把一个目录里的 `.txt` / `.md` 文件切成片段、建 TF-IDF 倒排索引并持久化；提问时按余弦相似度检索最相关片段，可选调用 OpenAI 兼容接口由 LLM 基于片段作答。

## 快速开始

```bash
python -m txt_rag build mydocs --out myidx.json
python -m txt_rag ask "GIL 是什么" --index myidx.json --top-k 3
```

## 无 API Key 如何运行

完全可以离线运行：检索逻辑是本地 TF-IDF 余弦相似度，不需要任何网络。不加 `--llm`（或未设置 `OPENAI_API_KEY`）时，工具会把检索到的片段直接拼在问题后面作为「答案」。

## 运行测试

```bash
python -m unittest discover -s tests -v
```

## License

MIT © ljiang9
