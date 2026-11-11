# 多模态自适应 RAG

面向教材、讲义、课件、扫描 PDF 与图片资料的多模态知识库。聊天入口使用 DeepSeek V4 Flash；Qwen3-VL Embedding 负责文本和图片的统一向量检索，Qwen3-VL Rerank 对召回结果排序；FastAPI 提供入库和流式问答接口，React 提供对话式工作台。

## 技术链路

```text
PDF / Word / PPT / Image
  → Unstructured 结构化解析
  → PaddleOCR 文本识别 + Qwen3-VL Plus 图表/图片描述
  → 置信度校验 + 父子分层切片
  → BM25 + Qwen3-VL 多模态稠密向量
  → RRF 统一融合 + MMR 去重
  → Qwen3-VL Rerank 精排
  → BERT 意图/难度分类 + LangChain Agent 查询改写、多 Query 扩展与拆解
  → 来源约束生成与幻觉检查
  → FastAPI / SSE → React
```

DeepSeek 主模型通过 `https://api.deepseek.com/chat/completions` 调用，密钥单独放在 `DEEPSEEK_API_KEY`；Qwen3-VL Embedding、Rerank 和图片描述通过百炼北京业务空间调用，使用 `BAILIAN_API_KEY` 与 `BAILIAN_WORKSPACE_ID`。所有密钥只从本地 `.env` 读取。运行前请先轮换在聊天中粘贴过的 API Key，再填入新密钥；仓库不包含真实密钥。图片素材保存在 `tmp/assets`，知识片段保存在 `RAG_KB_RUNTIME_PATH`。新模型使用独立的 Milvus 集合 `education_multimodal_qwen3_vl`，避免和旧向量空间混用。

## 目录结构

```text
manufacturing_rag/
  adapters/
    embedding.py         # 百炼 Qwen3-VL 多模态向量
    retriever.py         # BM25、多路融合、MMR、Milvus 候选融合
    rerank.py            # 百炼 Qwen3-VL 精排
    query_optimizer.py   # BERT + LangChain Agent 自适应查询
    ocr.py               # PaddleOCR 与百度 OCR 兼容适配
    vision.py            # Qwen3-VL Plus 图表/图片语义补全
    vector_store.py      # 本地 Docker Milvus 向量存储
  chunking.py            # 父子分层切片与低置信度校验
  etl.py                 # Unstructured 多格式 ETL 编排
  infrastructure.py      # Redis / MySQL
  pipeline.py            # 端到端 RAG 链路
  api.py                 # FastAPI、SSE 与入库接口
frontend/                # React + Vite 工作台
tests/                   # 单元与接口测试
Dockerfile
docker-compose.yml
```

## 本地开发

建议使用 Python 3.11。OCR 与文档解析依赖会占用较多安装空间。

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/install_nltk_data.py
Copy-Item .env.example .env
```

安装 NLTK 的 `punkt_tab` 和英文词性标注数据，供 Unstructured 解析 Word、PowerPoint 等 Office 文档使用，避免首次解析时尝试从受限的运行时地址下载。

在 `.env` 中分别设置新的 `DEEPSEEK_API_KEY`、`BAILIAN_API_KEY` 和 `BAILIAN_WORKSPACE_ID`。若已有旧 `.env`，将 `.env.example` 中模型配置段合并到本地文件；不要把密钥写入或提交到代码仓库。离线开发可临时使用：

向量库使用本机 Docker Milvus。仅启动向量数据库及其 etcd、MinIO 依赖（无需读取 `.env`）：

```powershell
docker compose -f docker-compose.vector.yml up -d
docker compose -f docker-compose.vector.yml ps
```

Milvus 地址为 `127.0.0.1:19530`，健康端点为 `http://127.0.0.1:9091/healthz`。宿主机运行 API 时使用 `MILVUS_ENDPOINT=http://127.0.0.1:19530`；主 `docker-compose.yml` 内使用 `http://milvus:19530`。OCR 在后端运行：Docker 部署由 API 容器中的 PaddleOCR 处理，宿主机开发则由本地 Python 后端处理。模型切换后如需重建当前知识库的索引，可在侧栏点击“重建资料索引”；该操作会重写新模型集合中的向量。

```powershell
$env:RAG_EMBEDDING_PROVIDER='hash'
$env:RAG_RERANK_PROVIDER='rule'
$env:RAG_LLM_PROVIDER='rule'
$env:RAG_QUERY_CLASSIFIER_PROVIDER='rule'
$env:RAG_QUERY_AGENT_PROVIDER='rule'
$env:RAG_VECTOR_STORE_PROVIDER='memory'
$env:RAG_REDIS_ENABLED='false'
$env:RAG_MYSQL_ENABLED='false'
py -m uvicorn manufacturing_rag.api:app --host 127.0.0.1 --port 8000 --reload
```

前端开发：

```powershell
Set-Location frontend
npm install
npm run dev
```

访问 `http://127.0.0.1:5173/`。前端通过 `POST /answer/stream` 消费 SSE 增量输出。

BERT 意图分类模型可用自建数据微调后写入默认目录：

```powershell
python scripts/train_intent_bert.py --dataset examples/intent_training.jsonl --output models/query-intent-bert
```

## Docker Compose 部署

复制环境配置后启动 API、React 静态资源、本地 Docker Milvus、Redis、MySQL、etcd 与 MinIO：

```powershell
Copy-Item .env.example .env
docker compose up -d --build
docker compose ps
```

服务地址为 `http://127.0.0.1:8000/frontend/`，健康检查为 `http://127.0.0.1:8000/health`。首次启动会下载本地模型；生产环境建议预先把微调后的 BERT 分类模型放在 `models/query-intent-bert`。

## 主要接口

| 接口 | 用途 |
| --- | --- |
| `GET /health` | 服务、模型、Redis、MySQL 和知识库状态 |
| `POST /answer` | 标准 JSON 问答 |
| `POST /answer/stream` | SSE 流式问答 |
| `POST /kb/document` | PDF、Word、PPT、文本、图片统一多模态 ETL |
| `POST /kb/pdf` | 兼容原 PDF 入库接口 |
| `POST /kb/image-ocr` | 图片 OCR；允许手动文本兜底 |
| `POST /kb/jsonl` | 结构化 JSONL 入库 |
| `GET /kb/chunks` | 查看子块、父块关联与置信度 |
| `POST /vector/ingest` | 按指定模式同步至 Milvus；默认追加 |
| `POST /vector/search` | 调试 Milvus 候选召回 |

统一文档入库请求示例：

```json
{
  "filename": "高等数学课件.pptx",
  "content_base64": "...",
  "mode": "append"
}
```

`append` 会保留其他文档；再次上传同名文档时只更新该文档。需要重建整库时显式传入 `"mode": "replace"`。

上传后的活动知识库会保存到 `RAG_KB_RUNTIME_PATH`，服务重启后自动恢复；清空向量集合不会删除这份本地知识库。

问答请求可以携带会话 ID：

```json
{"question":"极限的定义是什么？","session_id":"course-2026-user-001"}
```

## 数据约束

- 子块负责高精度召回，精排后恢复父块上下文，避免切片导致语义断裂。
- 原问题始终保留在 Multi-Query 召回列表中，改写/同义扩展结果通过 RRF 融合，避免改写丢失限定条件。
- BM25、稀疏向量和稠密向量只按排名做 RRF，不直接比较不同通道的原始分数。
- `RAG_BM25_WEIGHT`、`RAG_SPARSE_WEIGHT`、`RAG_DENSE_WEIGHT` 作为 RRF 路由权重使用。
- PDF 文本提取疑似乱码时优先转 OCR；OCR 不可用的异常页面会降置信度并标记 `needs_review`。
- 低于 `RAG_MIN_ETL_CONFIDENCE` 的内容保留并标记 `needs_review`，不会静默混入高质量语料。
- 生成只能使用检索来源；定义、公式、结论、步骤或答案缺少资料支撑时明确拒绝推断。
- 输出保留来源、页码、块号、模态、父块 ID 和相关性得分。
- Token、API Key、Secret、密码及未授权数据请求会被拦截或脱敏。

## 验证

```
powershell
pytest -q
python scripts/evaluate_materials.py
Set-Location frontend
npm run build
npm run dev
```

`evaluate_materials.py` 默认读取 `素材/*.pdf`，在离线确定性模式下逐条检查答案关键事实、来源页码/分数、幻觉校验和 JSON 响应字段；设置 `RAG_MATERIAL_EVAL_USE_RUNTIME=true` 可切换到当前运行时配置。

调试日志默认写入 `logs/rag_debug.jsonl`，记录 ETL、候选召回、重排和充足度判定，但不会记录明文凭据。



# Multimodal-knowledge-base

