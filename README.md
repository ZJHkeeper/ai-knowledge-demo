# ai-knowledge-demo

一个最小 RAG 示例项目：把 `data/` 里的 Markdown 文档写入本地 Chroma 向量库，然后从 Chroma 检索上下文并调用本地 Ollama 模型回答问题。

## 技术栈

- Python 3.13+：项目主语言，使用标准库 `argparse`、`pathlib`、`urllib`、`unittest` 等实现 CLI、文件读取、HTTP 调用和测试。
- ChromaDB：本地向量数据库，使用 `chromadb.PersistentClient` 将知识库 chunk 持久化到 `chroma_db/`。
- Chroma 默认 embedding：入库和检索时使用 Chroma collection 的默认 embedding 能力，首次运行可能会下载默认本地 embedding 模型。
- Ollama：本地大语言模型服务，问答阶段通过 `/api/chat` 调用，默认模型为 `qwen2.5:7b`。
- setuptools：通过 `pyproject.toml` 管理包构建、可编辑安装和命令行入口。
- unittest：项目测试框架，测试文件位于 `tests/`。

当前项目没有引入 LangChain，RAG 流程由项目代码直接串联 ChromaDB 和 Ollama。

## 项目架构

```text
ai-knowledge-demo/
├── data/                         # Markdown 知识库文档
│   └── refund_policy.md
├── chroma_db/                    # Chroma 本地持久化数据目录，入库后生成
├── src/
│   └── ai_knowledge_demo/
│       ├── cli.py                # 基础命令入口：ai-knowledge-demo
│       ├── ingest.py             # 文档发现、读取、切分、写入 Chroma
│       └── ask.py                # 检索、重排、调用 Ollama、格式化回答
├── tests/                        # 单元测试
├── pyproject.toml                # 项目元数据、依赖和命令行入口
└── README.md
```

核心流程分为两条命令链路：

```text
入库链路：
data/*.md
  -> discover_markdown_files()
  -> read_markdown_file()
  -> chunk_markdown()
  -> ingest_chunks()
  -> chroma_db/

问答链路：
用户问题
  -> generate_search_queries()
  -> retrieve_chunks()
  -> rerank_chunks()
  -> answer_question()
  -> Ollama /api/chat
  -> 带来源的中文回答
```

模块职责：

- `ai_knowledge_demo.ingest`：负责知识库构建。它递归发现 Markdown 文件，兼容 UTF-8 和 GB18030 编码读取内容，按标题、分隔线和长度切分 chunk，并写入 Chroma collection。
- `ai_knowledge_demo.ask`：负责 RAG 问答。它先让 Ollama 生成检索改写，再从 Chroma 扩大召回候选 chunk，通过关键词规则重排，最后把带来源标签的上下文提交给 Ollama 生成回答。
- `ai_knowledge_demo.cli`：基础 CLI 健康检查入口，安装后可通过 `ai-knowledge-demo` 调用。
- `tests/`：覆盖入库切分、CLI 和问答辅助逻辑，便于调整 chunk、检索和输出格式时做回归验证。

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

`--top-k` 表示最终交给 Ollama 的 chunk 数量。脚本内部会自动扩大 Chroma 候选召回，并用问题关键词对候选 chunk 重排，降低答案 chunk 被向量检索排到后面而漏掉的概率。

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

`ai_knowledge_demo.ingest` 会递归读取 `data/` 中的 `.md` 文件，按 Markdown 标题和分隔线优先切分，再按长度兜底切分。分隔线只作为 chunk 边界，不会作为独立内容写入 Chroma。每个 chunk 都会写入稳定 metadata，包括 `source`、`chunk_index`、`start_char` 和 `end_char`。

重复运行入库脚本时，会替换同一个 `source` 下的旧 chunk，避免文档更新后残留旧内容。向量生成使用 Chroma 默认 embedding，首次运行时 Chroma 可能会下载默认的本地 embedding 模型。

`ai_knowledge_demo.ask` 会复用入库模块的默认 Chroma 路径和 collection 名称，先用 Chroma `query()` 扩大召回候选，再用轻量关键词重排选出最终 top-k chunk，最后把带来源标签的上下文交给 Ollama `/api/chat` 生成中文回答。

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
