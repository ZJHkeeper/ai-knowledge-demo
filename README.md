# ai-knowledge-demo

A standard Python package scaffold for AI knowledge experiments.

## Setup

Create the virtual environment:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Run tests:

```powershell
.\.venv\Scripts\python -m unittest discover
```

Run the CLI module:

```powershell
.\.venv\Scripts\python -m ai_knowledge_demo.cli
```

Ingest Markdown files from `data/` into the local Chroma database:

```powershell
.\.venv\Scripts\python -m ai_knowledge_demo.ingest
```

## 设计思路

这个项目把 Markdown 知识文档作为最小数据源，默认从项目根目录的 `data/`
文件夹递归读取 `.md` 文件，再写入本地持久化 Chroma 向量数据库 `chroma_db/`。
入库脚本放在 `ai_knowledge_demo.ingest` 模块中，便于后续被 CLI、测试或其他
RAG 流程复用。

文档切分采用“先按 Markdown 结构分段，再按长度兜底切分”的策略。脚本会优先
根据标题行和 `---` 分隔线确定语义边界；如果某个段落超过默认 `800` 字符，
再按固定窗口切分，并保留 `100` 字符 overlap，降低上下文在 chunk 边界处丢失
的概率。

每个 chunk 都会写入稳定的 metadata，包括 `source`、`chunk_index`、
`start_char` 和 `end_char`。其中 `source` 使用相对于 `data/` 的路径，
方便追踪 chunk 来自哪份原始文档；字符范围则方便手动核对切分结果。

重复运行入库脚本时，会先删除同一 `source` 已有的 chunk，再写入新 chunk，
避免文档更新后旧内容残留。向量生成使用 Chroma 默认 embedding，因此无需额外
配置 API key；首次运行时 Chroma 可能会下载默认的本地 embedding 模型。

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
