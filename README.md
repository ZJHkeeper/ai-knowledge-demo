# ai-knowledge-demo

一个最小 RAG 示例项目：把 `data/` 里的 Markdown 文档写入本地 Chroma 向量库，然后从 Chroma 检索上下文并调用本地 Ollama 模型回答问题。

## Setup

创建虚拟环境：

```powershell
python -m venv .venv
```

激活虚拟环境：

```powershell
.\.venv\Scripts\Activate.ps1
```

安装项目依赖：

```powershell
.\.venv\Scripts\python -m pip install -e .
```

运行测试：

```powershell
.\.venv\Scripts\python -m unittest discover
```

## Ollama

安装并启动 Ollama 后，拉取默认问答模型：

```powershell
ollama pull qwen2.5:7b
```

确认 Ollama 正在运行：

```powershell
ollama list
```

默认连接地址是 `http://localhost:11434`，默认模型是 `qwen2.5:7b`。

## Ingest

将 `data/` 下的 Markdown 文件写入本地 Chroma 数据库：

```powershell
.\.venv\Scripts\python -m ai_knowledge_demo.ingest
```

也可以使用安装后的命令入口：

```powershell
ai-knowledge-ingest
```

默认配置：

- 数据目录：`data/`
- Chroma 持久化目录：`chroma_db/`
- Collection：`ai_knowledge_demo`
- Chunk 大小：`800`
- Chunk overlap：`100`

## Ask

从 Chroma 检索并调用本地 Ollama 回答：

```powershell
.\.venv\Scripts\python -m ai_knowledge_demo.ask "退款多久到账？"
```

也可以使用安装后的命令入口：

```powershell
ai-knowledge-ask "退款多久到账？"
```

可选参数：

```powershell
.\.venv\Scripts\python -m ai_knowledge_demo.ask "退款多久到账？" --top-k 4 --persist-dir chroma_db --collection ai_knowledge_demo --model qwen2.5:7b --ollama-url http://localhost:11434
```

如需换模型，可以直接传参数：

```powershell
.\.venv\Scripts\python -m ai_knowledge_demo.ask "退款多久到账？" --model llama3.1:8b
```

也可以用环境变量设置默认值：

```powershell
$env:OLLAMA_MODEL = "qwen2.5:7b"
$env:OLLAMA_URL = "http://localhost:11434"
```

回答会只基于检索到的知识库上下文生成，并在末尾列出来源，例如 `refund_policy.md#chunk=13`。如果知识库中没有相关上下文，会提示：`知识库中没有找到相关信息。`

## Design

`ai_knowledge_demo.ingest` 会递归读取 `data/` 中的 `.md` 文件，按 Markdown 标题和 `---` 分隔线优先切分，再按长度兜底切分。每个 chunk 都会写入稳定 metadata，包括 `source`、`chunk_index`、`start_char` 和 `end_char`。

重复运行入库脚本时，会替换同一个 `source` 下的旧 chunk，避免文档更新后残留旧内容。向量生成使用 Chroma 默认 embedding，首次运行时 Chroma 可能会下载默认的本地 embedding 模型。

`ai_knowledge_demo.ask` 会复用入库模块的默认 Chroma 路径和 collection 名称，先用 Chroma `query()` 检索 top-k chunk，再把带来源标签的上下文交给 Ollama `/api/chat` 生成中文回答。

手动查看切分结果可以直接调用 `chunk_markdown`：

```powershell
@'
from pathlib import Path
from ai_knowledge_demo.ingest import chunk_markdown, read_markdown_file

p = Path("data/refund_policy.md")
text = read_markdown_file(p)
chunks = chunk_markdown(text, p.name)

print("total chunks:", len(chunks))
for i, chunk in enumerate(chunks):
    print(f"\n--- chunk {i} [{chunk.metadata['start_char']}:{chunk.metadata['end_char']}] len={len(chunk.text)} ---")
    print(chunk.text)
'@ | .\.venv\Scripts\python -
```
