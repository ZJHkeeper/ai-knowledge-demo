# ai-knowledge-demo

一个最小 RAG 示例项目：把 `data/` 里的 Markdown、TXT、PDF、DOCX 文档切分为 chunk，同步写入本地 Chroma 向量库和 BM25 关键词索引，然后通过 query rewrite、混合检索、重排和本地 Ollama 模型完成带来源的问答，并用 eval 数据集做回归评估。

## 技术栈

- Python 3.13+：项目主语言，使用标准库 `argparse`、`pathlib`、`urllib`、`unittest` 等实现 CLI、文件读取、HTTP 调用和测试。
- ChromaDB：本地向量数据库，使用 `chromadb.PersistentClient` 将知识库 chunk 持久化到 `chroma_db/`。
- Chroma 默认 embedding：入库和检索时使用 Chroma collection 的默认 embedding 能力，首次运行可能会下载默认本地 embedding 模型。
- rank-bm25：关键词检索组件，入库后生成持久化 BM25 索引，问答和评估时与 Chroma 向量检索做混合召回。
- pypdf / python-docx：入库阶段提取 PDF 和 DOCX 文档中的文本内容。
- Ollama：本地大语言模型服务，问答阶段通过 `/api/chat` 调用，默认模型为 `qwen2.5:7b`。
- setuptools：通过 `pyproject.toml` 管理包构建、可编辑安装和命令行入口。
- unittest：项目测试框架，测试文件位于 `tests/`。

当前项目没有引入 LangChain，RAG 流程由项目代码直接串联 ChromaDB、BM25 和 Ollama。

## 项目架构

```text
ai-knowledge-demo/
├── data/                         # Markdown、TXT、PDF、DOCX 知识库文档
│   ├── refund_policy.md
│   ├── logistics_exception_rules.txt
│   ├── customer_service_manual.md
│   ├── payment_and_membership_rules.md
│   ├── refund_edge_cases.md
│   ├── international_payment_refund_manual.pdf
│   └── membership_invoice_rules.docx
├── chroma_db/                    # Chroma 本地持久化目录，入库后也会生成 bm25_index.json
├── src/
│   └── ai_knowledge_demo/
│       ├── cli.py                # 基础命令入口：ai-knowledge-demo
│       ├── ingest.py             # 文档发现、读取、切分、写入 Chroma 和 BM25 索引
│       ├── ask.py                # query rewrite、混合检索、重排、调用 Ollama
│       └── evaluate.py           # 批量评估 eval case 的检索和回答质量
├── tests/                        # 单元测试和 eval 数据集
│   └── eval_cases.json
├── pyproject.toml                # 项目元数据、依赖和命令行入口
└── README.md
```

核心流程分为三条命令链路：

```text
入库链路：
data/*.{md,txt,pdf,docx}
  -> discover_document_files()
  -> read_document_file()
  -> chunk_markdown()
  -> ingest_chunks()
  -> write_bm25_index()
  -> chroma_db/
  -> chroma_db/bm25_index.json

问答链路：
用户问题
  -> generate_search_queries()
  -> retrieve_chunks()  # Chroma 向量检索 + BM25 关键词检索
  -> rerank_chunks()
  -> answer_question()
  -> Ollama /api/chat
  -> 带来源的中文回答

评估链路：
tests/eval_cases.json
  -> evaluate_cases()
  -> evaluate_case()
  -> retrieve_chunks()
  -> answer_question()  # --retrieval-only 时跳过
  -> 检索来源和回答断言结果
```

模块职责：

- `ai_knowledge_demo.ingest`：负责知识库构建。它递归发现 `.md`、`.txt`、`.pdf`、`.docx` 文件，将文档解析为文本后按标题、分隔线和长度切分 chunk，并同步写入 Chroma collection 和 BM25 索引。
- `ai_knowledge_demo.ask`：负责 RAG 问答。它先让 Ollama 生成检索改写，再执行 Chroma 向量召回和 BM25 关键词召回，合并去重后做 hybrid rerank，最后把带来源标签的上下文提交给 Ollama 生成回答。
- `ai_knowledge_demo.evaluate`：负责批量评估。它读取 `tests/eval_cases.json`，复用问答链路做检索和可选回答生成，并检查期望来源、必含内容、建议内容和拒答预期。
- `ai_knowledge_demo.cli`：基础 CLI 健康检查入口，安装后可通过 `ai-knowledge-demo` 调用。
- `tests/`：覆盖入库切分、CLI、问答辅助逻辑和评估入口，便于调整 chunk、检索、重排和输出格式时做回归验证。

## 问题与优化记录

本节记录项目迭代过程中每次提问解决的问题、排查过程和最终沉淀的优化方案，方便后续回看项目为什么这样设计。

### 设计 Markdown 入库脚本

- 问题：需要把 `data/` 文件夹中的 `.md` 文档切分成 chunk，并写入 Chroma 向量数据库。
- 解决过程：先检查项目结构、`pyproject.toml`、现有 CLI 和测试，确认项目是轻量 Python 包且还没有入库能力；再确认 embedding 方案，最终选择 Chroma 默认 embedding，避免额外接入 OpenAI API 或独立本地模型配置。
- 结论：入库流程应尽量简单透明，默认从项目根目录 `data/` 读取 Markdown，写入本地 `chroma_db/`，并保留可追踪的 metadata。
- 优化方案：新增 `ai_knowledge_demo.ingest`，实现文档发现、编码兼容读取、Markdown chunk 切分、Chroma 持久化写入和重复入库替换逻辑；同时增加单元测试覆盖短文本、长文本 overlap 和 metadata 稳定性。
- 示例输出：运行入库命令后，可以看到读取文件数、写入 chunk 数和 Chroma 持久化位置。

```text
> ai-knowledge-ingest
Files read: 1
Chunks written: 16
Collection: ai_knowledge_demo
Chroma path: D:\CodexProjects\ai-knowledge-demo\chroma_db
```

手动打印前两个 chunk 时，可以看到每个 chunk 都带有可追踪 metadata：

```text
chunk=0 source=refund_policy.md start=0 end=9
# 退款与售后政策

chunk=1 source=refund_policy.md start=11 end=90
## 一、退款申请时效
...
```

### 优化 Markdown 分隔线切分

- 问题：手动查看 chunk 时发现存在单独内容为 `---` 的 chunk，例如 `--- chunk 14 [655:658] len=3 ---`，这类 chunk 没有检索语义，会污染向量库。
- 解决过程：先统计样例文档中的分隔线 chunk，确认 `refund_policy.md` 会生成 14 个仅包含 `---` 的 chunk；随后把处理策略从“分隔线作为普通 span”改为“分隔线只作为边界”，并扩展识别 `---`、`***`、`___`、`- - -` 等 Markdown thematic break 写法。
- 结论：Markdown 分隔线应保留为章节边界，但不应作为独立文档内容写入 Chroma。
- 优化方案：更新 `_markdown_spans`，让标题继续作为新 chunk 起点，分隔线只断开前后内容且自身不产出 chunk；同步调整测试和 README，并重新 ingest，使 Chroma 中旧的分隔线 chunk 被同一 `source` 的替换逻辑清除。
- 示例输出：优化前手动查看 chunk 时会出现只有分隔线的无意义内容。

```text
--- chunk 14 [655:658] len=3 ---
---
```

优化后再打印切分结果，分隔线只作为边界，输出中不再出现正文为 `---` 的 chunk：

```text
--- chunk 2 [97:195] len=98 ---
## 二、退款到账时间
...

--- chunk 3 [202:311] len=109 ---
## 三、不可退款情况
...
```

### 新增知识库问答脚本

- 问题：入库后还缺少一个脚本，能够从 Chroma 检索知识片段，并调用模型生成回答。
- 解决过程：先确认 `ingest.py` 中已有默认 collection、持久化目录和 metadata 设计，再新增与 `ingest.py` 同级的 `ask.py`，复用这些默认常量，避免问答脚本和入库脚本各自维护一套配置。
- 结论：问答脚本应放在包内 `src/ai_knowledge_demo/ask.py`，支持 `python -m ai_knowledge_demo.ask "问题"` 和安装后的 `ai-knowledge-ask` 命令入口。
- 优化方案：实现 Chroma 查询、上下文格式化、模型调用、来源输出和单元测试；`--top-k`、`--persist-dir`、`--collection` 等参数用于控制检索范围和目标 collection。
- 示例输出：执行问答命令后，终端先打印检索 query，再打印三段式回答和来源正文。

```text
> ai-knowledge-ask "退款多久到账？"

结论：审核通过后，款项通常会在 3-5 个工作日内原路返回。

来源：
- refund_policy.md#chunk=2
## 二、退款到账时间
...
```

### 从 OpenAI 切换为本地 Ollama

- 问题：OpenAI API 需要 API key 和可用额度，实际运行时出现 `insufficient_quota`，不适合做一个无需付费额度的本地 demo。
- 解决过程：先确认 OpenAI 调用已经成功发起，问题来自账号额度而不是代码；随后比较替代方案，选择 Ollama 作为本地模型服务，避免云端 API key 和计费依赖。
- 结论：这个项目更适合作为本地 RAG demo，模型生成阶段使用 Ollama `/api/chat`，默认模型为 `qwen2.5:7b`。
- 优化方案：移除 OpenAI SDK 依赖和 `OPENAI_API_KEY` 要求，改用标准库 `urllib` 调用 Ollama；README 增加 Ollama 安装、拉取模型和运行命令说明。
- 示例输出：切换前如果 OpenAI 账号额度不可用，问答阶段会失败。

```text
OpenAI request failed: insufficient_quota
```

切换后，模型调用改为本地 Ollama。只要本机服务可用，问答命令会继续输出本地模型生成的回答：

```text
> ai-knowledge-ask "退款多久到账？" --model qwen2.5:7b --ollama-url http://localhost:11434

结论：审核通过后，款项通常会在 3-5 个工作日内原路返回。
```

### 修复“退款多久到账？”漏召回答案 chunk

- 问题：用户询问“退款多久到账？”时，知识库中明明有“退款到账时间”chunk，但默认 top-k 检索没有把它放进上下文，模型因此回答“知识库中没有找到相关信息”。
- 解决过程：查询 Chroma 完整排名后发现答案 chunk 排在第 9，而默认 `top-k=4`，根因是 Chroma 默认 embedding 对中文语义检索不够稳定，而不是 chunk 切分问题。
- 结论：应先扩大召回候选，再做轻量重排，不能直接把 Chroma 原始前 4 条交给模型。
- 优化方案：内部召回数量改为 `min(collection.count(), max(top_k * 5, 20))`，再通过中文关键词和字符 n-gram 对候选 chunk 重排，最终仍只取用户指定的 top-k 给 Ollama。
- 示例输出：优化前只取 Chroma 原始前 4 条时，真正答案 chunk 可能排在后面。

```text
原始 Chroma 排名：
1. refund_policy.md#chunk=7  发票相关说明
2. refund_policy.md#chunk=10 特殊商品说明
3. refund_policy.md#chunk=3  不可退款情况
4. refund_policy.md#chunk=5  优惠券退款规则
...
9. refund_policy.md#chunk=2  退款到账时间
```

优化后先扩大召回，再按“退款”“到账”等关键词重排，最终传给 Ollama 的上下文会包含答案 chunk：

```text
重排后 top-k：
1. refund_policy.md#chunk=2  退款到账时间
2. refund_policy.md#chunk=3  不可退款情况
3. refund_policy.md#chunk=5  优惠券退款规则
4. refund_policy.md#chunk=7  发票相关说明
```

### 放宽过度严格的回答策略

- 问题：询问“我可以开具个人发票吗？”时，检索已经命中发票相关 chunk，但知识库没有明确写“个人发票”，模型仍直接回答“没有找到相关信息”。
- 解决过程：检查上下文后确认发票 chunk 包含“电子发票”和“企业用户可申请增值税专用发票”，属于“相关但未明确说明”而不是“完全无关”。
- 结论：回答策略不能只有“能回答”和“不知道”两档，否则会掩盖相近信息。
- 优化方案：调整 prompt，要求模型在相关但不完全匹配时说明“知识库中有相关信息，但未明确说明……”，并列出可确认内容。
- 示例输出：优化前，模型容易把“相关但未明确”的问题直接判成无信息。

```text
结论：知识库中没有找到相关信息。
```

优化后，回答会保留相关事实，并明确指出缺口：

```text
结论：知识库中有发票相关信息，但未明确说明是否可以开具个人发票。
可以确认：
- 电子发票将在支付成功后 1 小时内生成。
- 企业用户可申请增值税专用发票。
未明确说明：
- 未明确说明个人用户是否可以开具个人发票。
```

### 引入三档 answerability 和固定输出模板

- 问题：放宽 prompt 后，本地模型有时仍会输出不稳定格式，或者把“有相关信息但未明确”误写成“知识库中没有找到相关信息”。
- 解决过程：把回答策略明确为三档：`可直接回答`、`部分相关但未明确`、`完全无关`；同时要求固定输出 `结论`、`可以确认`、`未明确说明` 三个字段，并增加轻量后处理兜底。
- 结论：本地模型输出需要模板约束和最小后处理共同保证稳定性。
- 优化方案：在 `build_ollama_payload` 中写入三档 answerability 规则；新增 `normalize_answer_template`，当上下文明显相关但模型误用 no-context 结论时自动修正，并保证三段模板字段存在。
- 示例输出：固定模板后，即使知识库完全无关，也会稳定输出同样的字段结构。

```text
> ai-knowledge-ask "是否支持货到付款？"
结论：知识库中没有找到相关信息。
可以确认：
- 无
未明确说明：
- 用户问题相关内容未在知识库中出现。
```

如果模型漏掉字段，`normalize_answer_template()` 会补齐 `结论`、`可以确认`、`未明确说明` 三段，避免 CLI 输出格式漂移。

### 用多查询改写解决语义表达不一致

- 问题：用户问“App 显示商品已到达，我没有收到商品”，知识库写的是“商品在运输过程中丢失”，两者语义接近但字面关键词不同，导致物流 chunk 召回排名靠后。
- 解决过程：先查看 Chroma 原始排名和重排结果，发现物流 chunk 只排到较后位置；再让 Ollama 生成多条检索改写 query，把口语场景扩展为“物流异常、运输丢失、未收货、客服、补发、全额退款”等知识库可能使用的政策词。
- 结论：不要为每个问题硬编码同义词，更通用的方式是 query rewrite / query expansion，让检索阶段覆盖多种表达。
- 优化方案：新增 `generate_search_queries`、`build_query_rewrite_payload` 和 `parse_search_queries`；每条 query 分别查 Chroma，结果合并去重，再用原问题和改写 query 的累计关键词分数重排。
- 示例输出：用户原问题和知识库措辞不一致时，先由 Ollama 生成多条检索 query。

```text
> ai-knowledge-ask "App 显示商品已到达，我没有收到商品"
检索查询：
- App 显示商品已到达，我没有收到商品
- 物流显示已送达但未收到
- 运输途中丢失处理
- 未收货 联系客服 补发 全额退款
```

这些 query 会让“物流相关问题”chunk 进入候选：

```text
来源：
- refund_policy.md#chunk=4
## 四、物流相关问题

若商品在运输过程中丢失，用户可联系客服重新补发或申请全额退款。
```

### 降低泛词对重排的干扰

- 问题：多查询改写后，`商品`、`处理`、`显示` 等泛词会让“特殊商品说明”这类 chunk 得分过高，仍可能排在真正的“物流相关问题”之前。
- 解决过程：分析各 chunk 的关键词分数后发现，低信号泛词造成多个 chunk 打平，向量距离 tie-breaker 又把不相关 chunk 排前。
- 结论：关键词重排需要过滤低信号词，让更具体的业务词决定排序。
- 优化方案：新增 `LOW_SIGNAL_TERMS`，过滤 `商品`、`处理`、`显示`、`用户`、`问题` 等泛词；重排由单 query 最高分改为多 query 累积分数，使同时命中多个业务术语的 chunk 排名更高。
- 示例输出：优化前，泛词命中会让不够相关的 chunk 得分偏高。

```text
关键词命中：
refund_policy.md#chunk=10 特殊商品说明  命中：商品
refund_policy.md#chunk=4  物流相关问题  命中：商品、运输丢失、未收货、客服、补发、全额退款
```

过滤低信号词并累计多 query 得分后，更具体的物流 chunk 会排到前面：

```text
重排后 top-k：
1. refund_policy.md#chunk=4  物流相关问题
2. refund_policy.md#chunk=10 特殊商品说明
```

### 支持多格式测试文档入库

- 问题：`data/` 中新增了 TXT、PDF、DOCX、MD 多种格式的测试文档，但入库脚本原本只递归发现和读取 `.md` 文件，导致其它格式不会进入 Chroma 知识库。
- 解决过程：先确认现有入库链路已经把“切分”和“写入 Chroma”封装清楚，因此不需要重写 chunk 逻辑；改造重点放在文件发现和文档解析层，并为 PDF/DOCX 引入专用解析依赖。随后检查样例 `international_payment_refund_manual.pdf`、`logistics_exception_rules.txt` 和 `membership_invoice_rules.docx` 的抽取结果，发现三者都缺少 `#` 标题标记，短文档会被保存为单个 chunk。
- 结论：入库脚本应统一把支持的文档格式转换为纯文本，再复用现有 `chunk_markdown()`、metadata 和同源替换逻辑；非 Markdown 文档抽取后还需要做轻量结构化，把页码、页边界和常见章节标题恢复成切分器能识别的文本结构。
- 优化方案：新增 `discover_document_files()` 和 `read_document_file()`，支持 `.md`、`.txt`、`.pdf`、`.docx`；PDF 通过 `pypdf` 提取文本层，DOCX 通过 `python-docx` 提取正文段落和表格文本；TXT、PDF、DOCX 统一将 `一、...`、`1. ...` 等章节行转换为 `## ...`，PDF 额外过滤 `page 1` / `第 1 页` 这类页码行并用 Markdown 分隔线标记页边界；`.md` 保持原文读取，避免改变 Markdown 空行和格式。
- 验证结果：新增单元测试覆盖多扩展名发现、文本编码回退、DOCX 段落和表格提取、PDF 文本提取，以及 TXT、DOCX、PDF 章节切分；真实入库可以读取 7 个样例文档，写入 77 个 chunk。

```text
logistics_exception_rules.txt
chunks: 1 -> 6

membership_invoice_rules.docx
chunks: 1 -> 7

international_payment_refund_manual.pdf
chunks: 1 -> 7

> .\.venv\Scripts\python -m unittest discover
Ran 28 tests
OK

> .\.venv\Scripts\python -m ai_knowledge_demo.ingest
Files read: 7
Chunks written: 77
Collection: ai_knowledge_demo
Chroma path: D:\CodexProjects\ai-knowledge-demo\chroma_db
```

### 引入混合搜索与检索分数输出

- 问题：当前问答链路主要依赖 Chroma 向量搜索，虽然已经通过扩大召回、query rewrite 和关键词重排改善了“退款多久到账？”这类语义检索问题，但对“Visa 退款需要多久？”这类包含精确实体的问题仍不够稳定。`Visa` 可能命中“支付方式支持”段落，而“退款需要多久”又需要命中“退款到账时间”或“国际信用卡退款”段落，单一路径容易漏掉其中一类证据。
- 解决过程：先确认知识库中 `Visa`、`信用卡`、`国际银行卡`、`退款到账时间` 分散在多个 chunk 中，再引入 BM25 关键词检索作为 Chroma 向量检索的补充。入库阶段为每个 chunk 生成持久化 BM25 索引；问答阶段同时执行向量召回和 BM25 召回，合并去重后统一重排。
- 结论：精确实体类问题不应只依赖向量相似度。向量检索负责语义召回，BM25 负责品牌、支付方式、时间范围等字面关键词召回，两者混合后更适合政策问答场景。
- 优化方案：新增 `rank-bm25` 依赖；`ingest.py` 在写入 Chroma 后同步生成 `chroma_db/bm25_index.json`，索引包含 chunk id、正文、metadata 和 tokens；`ask.py` 默认启用混合搜索，并支持 `--bm25-index` 指定索引路径，支持用 `--no-bm25`、`--no-vector` 和 `--no-rerank` 分别关闭 BM25、向量检索和重排。分词策略保留英文/数字词，并为中文生成 2-3 字 n-gram，同时对 `Visa`、`MasterCard` 扩展出 `信用卡`、`国际信用卡`、`国际银行卡` 等业务词。
- 分数输出：来源区新增每个 chunk 的 `向量权重分` 和 `关键词权重分`，用于观察最终上下文为什么被选中。向量权重分来自向量距离转换后的加权贡献，关键词权重分来自 BM25 分数归一化后的加权贡献。
- 验证结果：`Visa 退款需要多久？` 的默认 top-k 可同时召回 `Visa 信用卡` 支付方式、国际信用卡退款时间、普通退款到账时间等来源；测试覆盖 BM25 索引生成、缺失索引回退、混合重排和来源分数输出。
```text
> .\.venv\Scripts\python -m unittest discover
Ran 36 tests
OK

来源：
- refund_policy.md#chunk=8 | 向量权重分=1.019 | 关键词权重分=10.000
## 八、支付方式支持
...

- refund_policy.md#chunk=2 | 向量权重分=0.998 | 关键词权重分=2.510
## 二、退款到账时间
...
```

### 新增 eval 数据集与评估脚本

- 问题：现有单元测试能覆盖切分、检索重排、输出格式等局部逻辑，但缺少一组面向真实用户问题的端到端评估数据。每次调整 query rewrite、BM25、重排或回答策略后，只靠手工提问不容易稳定判断整体效果是否退化。
- 解决过程：新增 `tests/eval_cases.json` 作为 eval 数据集，按业务场景记录问题、类型说明、可接受来源、回答必须包含内容、建议包含内容和是否预期拒答。新增包内评估入口 `ai_knowledge_demo.evaluate`，批量读取 eval case，复用现有问答链路执行 query rewrite、混合检索和重排，并按 case 语义校验检索来源与回答内容。
- 结论：eval 应默认模拟真实用户提问流程，而不是只做原始问题直检。因此即使使用 `--retrieval-only`，默认仍会执行 query rewrite，只跳过最终回答生成；如需定位 query rewrite 的贡献，可显式传入 `--no-query-rewrite` 做消融对照。
- 优化方案：新增安装入口 `ai-knowledge-evaluate`；评估脚本支持 `--cases`、`--top-k`、`--persist-dir`、`--collection`、`--bm25-index`、`--no-bm25`、`--no-vector`、`--no-rerank`、`--model`、`--ollama-url`、`--retrieval-only` 和 `--no-query-rewrite`。`expected_sources` 表示可接受来源集合，命中任意一个即通过；`must_include` 为硬性答案断言；`should_include` 只输出 warning，不影响通过率；`expected_refusal` 用于区分应拒答和不应拒答场景。
- 验证结果：默认 retrieval-only eval 会经过 query rewrite，10 条 eval case 全部通过；关闭 query rewrite 后，`EVAL-002`（“钱什么时候能退回来？”）无法稳定命中 `refund_policy.md#chunk=2`，说明该 case 有效覆盖了语义表达不一致时 query rewrite 的价值。
```text
> .\.venv\Scripts\python -m unittest discover
Ran 50 tests
OK

> .\.venv\Scripts\python -m ai_knowledge_demo.evaluate --retrieval-only
Summary: 10/10 passed (100.0%).

> .\.venv\Scripts\python -m ai_knowledge_demo.evaluate --retrieval-only --no-query-rewrite
Summary: 9/10 passed (90.0%).
```

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

`ai_knowledge_demo.ingest` 用于构建本地知识库索引。它会发现 `data/` 下支持的文档，提取文本并切分为 chunk，然后同时写入 Chroma 向量库和 BM25 关键词索引。后续 `ask` 和 `evaluate` 会复用这两份索引做混合检索。

默认入库链路会递归读取 `.md`、`.txt`、`.pdf`、`.docx` 文件；其中 `.md` 和 `.txt` 会按 UTF-8/GB18030 兼容读取，`.pdf` 提取文本层内容，`.docx` 提取正文段落和表格文本。文本会按 Markdown 标题和分隔线优先切分，再按 `--chunk-size` 和 `--chunk-overlap` 兜底切分。每个 chunk 会写入稳定 metadata，包括来源文件、chunk 序号和字符范围。

入库完成后会生成两类检索资产：

- Chroma collection：用于向量检索，默认持久化在 `chroma_db/`。
- BM25 index：用于关键词检索，默认写入 `<persist-dir>/bm25_index.json`，内容包含 chunk 正文、metadata 和分词 tokens。

常用命令：

```powershell
# 使用默认配置入库，并同步生成 Chroma 和 BM25 索引
.\.venv\Scripts\python -m ai_knowledge_demo.ingest
```

```powershell
# 使用安装后的命令入口
ai-knowledge-ingest
```

```powershell
# 指定数据目录、Chroma 目录、collection 和 BM25 索引路径
.\.venv\Scripts\python -m ai_knowledge_demo.ingest --data-dir data --persist-dir chroma_db --collection ai_knowledge_demo --bm25-index chroma_db\bm25_index.json
```

```powershell
# 调整切分参数
.\.venv\Scripts\python -m ai_knowledge_demo.ingest --chunk-size 800 --chunk-overlap 100
```

运行成功后会输出读取文件数、写入 chunk 数、collection、Chroma 路径和 BM25 索引路径。

ingest CLI 可控参数：

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--data-dir` | `data/` | 指定待入库文档目录，会递归发现支持的文档文件。 |
| `--persist-dir` | `chroma_db/` | 指定 Chroma 本地持久化目录。 |
| `--collection` | `ai_knowledge_demo` | 指定写入的 Chroma collection 名称。 |
| `--bm25-index` | `<persist-dir>/bm25_index.json` | 指定 BM25 索引输出路径。 |
| `--chunk-size` | `800` | 指定 chunk 目标长度。 |
| `--chunk-overlap` | `100` | 指定长文本兜底切分时相邻 chunk 的重叠长度。 |

## Ask

`ai_knowledge_demo.ask` 用于对本地知识库发起一次 RAG 问答。它会读取已经入库到 Chroma 的 chunk，结合 query rewrite、向量检索、BM25 检索和 hybrid rerank 选出最终上下文，再调用本地 Ollama 生成带来源的中文回答。

默认问答链路会先让 Ollama 生成多条检索 query，然后从 Chroma 扩大召回向量候选，并追加 BM25 关键词候选；候选合并去重后会根据关键词、BM25 分数和向量距离做重排，最后只把 `--top-k` 指定数量的 chunk 交给 Ollama。普通 ask CLI 当前默认执行 query rewrite，未暴露关闭开关；如果需要做 rewrite 消融，使用 `evaluate --no-query-rewrite` 更合适。

常用命令：

```powershell
# 使用默认配置问答
.\.venv\Scripts\python -m ai_knowledge_demo.ask "退款多久到账？"
```

```powershell
# 使用安装后的命令入口
ai-knowledge-ask "退款多久到账？"
```

```powershell
# 指定最终上下文数量、知识库位置、collection 和 Ollama 配置
.\.venv\Scripts\python -m ai_knowledge_demo.ask "退款多久到账？" --top-k 4 --persist-dir chroma_db --collection ai_knowledge_demo --model qwen2.5:7b --ollama-url http://localhost:11434
```

```powershell
# 关闭 BM25，观察纯向量检索问答效果
.\.venv\Scripts\python -m ai_knowledge_demo.ask "退款多久到账？" --no-bm25

# 关闭向量检索，观察纯 BM25 问答效果
.\.venv\Scripts\python -m ai_knowledge_demo.ask "退款多久到账？" --no-vector

# 关闭 hybrid rerank，保留检索结果合并后的原始顺序
.\.venv\Scripts\python -m ai_knowledge_demo.ask "退款多久到账？" --no-rerank
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

回答会先打印本次使用的检索 query，再基于检索到的知识库上下文生成三段式回答，并在末尾列出来源正文，例如 `refund_policy.md#chunk=13`。如果知识库中没有相关上下文，会提示：`知识库中没有找到相关信息。`

ask CLI 可控参数：

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `question` | 必填 | 用户问题，会用于 query rewrite、检索重排和最终回答生成。 |
| `--top-k` | `4` | 指定最终交给 Ollama 的 chunk 数量。 |
| `--persist-dir` | `chroma_db` | 指定 Chroma 本地持久化目录。 |
| `--collection` | `ai_knowledge_demo` | 指定 Chroma collection 名称。 |
| `--bm25-index` | `<persist-dir>/bm25_index.json` | 指定 BM25 索引文件路径。 |
| `--no-bm25` | BM25 开启 | 关闭 BM25 关键词检索，仅保留向量检索。 |
| `--no-vector` | 向量检索开启 | 关闭 Chroma 向量检索，仅保留 BM25 检索。 |
| `--no-rerank` | hybrid rerank 开启 | 关闭重排，保留检索结果合并去重后的原始顺序。 |
| `--model` | `OLLAMA_MODEL` 或 `qwen2.5:7b` | 指定 query rewrite 和回答生成使用的 Ollama 模型。 |
| `--ollama-url` | `OLLAMA_URL` 或 `http://localhost:11434` | 指定 Ollama 服务地址。 |

## Evaluate

`ai_knowledge_demo.evaluate` 用于批量评估本地 RAG 链路在 eval 数据集上的检索和回答效果。默认数据集是 `tests/eval_cases.json`，每条 case 会记录用户问题、可接受来源、回答必须包含内容、建议包含内容以及是否预期拒答。

默认评估会尽量模拟真实问答流程：先执行 query rewrite，再执行 Chroma 向量检索和 BM25 检索，合并去重后做 hybrid rerank，最后调用 Ollama 生成回答并检查断言。加上 `--retrieval-only` 后只检查检索来源，不生成最终回答；query rewrite、向量检索、BM25 和 rerank 仍默认开启，适合做检索质量回归。

常用命令：

```powershell
# 只评估检索来源，速度更快，适合回归检索质量
.\.venv\Scripts\python -m ai_knowledge_demo.evaluate --retrieval-only

# 关闭 query rewrite 做消融
.\.venv\Scripts\python -m ai_knowledge_demo.evaluate --retrieval-only --no-query-rewrite

# 关闭 BM25，观察纯向量检索效果
.\.venv\Scripts\python -m ai_knowledge_demo.evaluate --retrieval-only --no-bm25

# 关闭向量检索，观察纯 BM25 效果
.\.venv\Scripts\python -m ai_knowledge_demo.evaluate --retrieval-only --no-vector

# 关闭 hybrid rerank，观察原始合并顺序效果
.\.venv\Scripts\python -m ai_knowledge_demo.evaluate --retrieval-only --no-rerank
```

eval CLI 可控开关：

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--cases` | `tests/eval_cases.json` | 指定 eval case JSON 文件路径。 |
| `--top-k` | `4` | 指定最终用于校验或回答的 chunk 数量。 |
| `--persist-dir` | `chroma_db` | 指定 Chroma 本地持久化目录。 |
| `--collection` | `ai_knowledge_demo` | 指定 Chroma collection 名称。 |
| `--bm25-index` | `<persist-dir>/bm25_index.json` | 指定 BM25 索引文件路径。 |
| `--no-bm25` | BM25 开启 | 关闭 BM25 关键词检索，仅保留向量检索。 |
| `--no-vector` | 向量检索开启 | 关闭 Chroma 向量检索，仅保留 BM25 检索。 |
| `--no-rerank` | hybrid rerank 开启 | 关闭重排，保留检索结果合并去重后的原始顺序。 |
| `--model` | `OLLAMA_MODEL` 或 `qwen2.5:7b` | 指定 query rewrite 和回答生成使用的 Ollama 模型。 |
| `--ollama-url` | `OLLAMA_URL` 或 `http://localhost:11434` | 指定 Ollama 服务地址。 |
| `--retrieval-only` | 关闭 | 只校验检索来源，不调用 `answer_question()` 生成最终回答。 |
| `--no-query-rewrite` | query rewrite 开启 | 关闭 query rewrite，仅使用原始问题检索。 |

## Design

`ai_knowledge_demo.ingest` 会递归读取 `data/` 中的 `.md`、`.txt`、`.pdf`、`.docx` 文件，将内容转换为纯文本后按 Markdown 标题和分隔线优先切分，再按长度兜底切分。分隔线只作为 chunk 边界，不会作为独立内容写入 Chroma。每个 chunk 都会写入稳定 metadata，包括 `source`、`chunk_index`、`start_char` 和 `end_char`。

重复运行入库脚本时，会替换同一个 `source` 下的旧 chunk，避免文档更新后残留旧内容。向量生成使用 Chroma 默认 embedding，首次运行时 Chroma 可能会下载默认的本地 embedding 模型。入库脚本还会基于当前 chunk 全量生成 BM25 索引文件，默认路径为 `chroma_db/bm25_index.json`；索引保存正文、metadata 和混合中英文 tokens，供后续关键词召回使用。

`ai_knowledge_demo.ask` 会复用入库模块的默认 Chroma 路径、collection 名称和 BM25 索引路径。默认链路先让 Ollama 生成多条检索 query，再用 Chroma `query()` 扩大召回向量候选，同时用 BM25 索引召回关键词候选；候选合并去重后通过 hybrid rerank 选出最终 top-k chunk。重排同时考虑关键词命中、BM25 归一化分数和向量距离，最后把带来源标签的上下文交给 Ollama `/api/chat` 生成三档 answerability 的中文回答。`--no-bm25`、`--no-vector` 和 `--no-rerank` 可用于关闭对应检索或重排步骤。

`ai_knowledge_demo.evaluate` 复用同一套检索和回答函数批量运行 `tests/eval_cases.json`。它默认模拟真实问答链路，也就是 query rewrite、向量检索、BM25、rerank 和回答生成都会执行；`--retrieval-only` 会跳过最终回答生成，只校验检索来源；`--no-query-rewrite`、`--no-bm25`、`--no-vector` 和 `--no-rerank` 可用于做消融对照。

手动查看切分结果可以直接调用 `chunk_markdown`：

```powershell
@'
from pathlib import Path
from ai_knowledge_demo.ingest import chunk_markdown, read_document_file

p = Path("data/refund_policy.md")
text = read_document_file(p)
chunks = chunk_markdown(text, p.name)

print("total chunks:", len(chunks))
for i, chunk in enumerate(chunks):
    print(f"\n--- chunk {i} [{chunk.metadata['start_char']}:{chunk.metadata['end_char']}] len={len(chunk.text)} ---")
    print(chunk.text)
'@ | .\.venv\Scripts\python -
```
