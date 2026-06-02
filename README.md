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
