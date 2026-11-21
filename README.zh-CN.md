# 多模态知识库

[English](README.md) | [简体中文](README.zh-CN.md)

![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white) [![MIT 许可证](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> 面向教育资料的多模态知识库，结合文档和图片导入、混合检索与可调用工具的 RAG Agent，通过 FastAPI 服务和 React 界面提供问答能力。

## 目录

- [功能](#功能)
- [工作流程](#工作流程)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [配置与数据](#配置与数据)
- [API 概览](#api-概览)
- [开发](#开发)
- [安全与隐私](#安全与隐私)
- [使用限制](#使用限制)
- [许可证](#许可证)

## 功能

- 支持导入 PDF、Word、PowerPoint、文本、Markdown 和图片。
- 使用 Unstructured 解析文档、PaddleOCR 识别文字，并通过 Qwen3-VL 生成图片和图表描述。
- 结合 BM25 与稠密向量检索，对候选结果进行融合、多样化处理和重排。
- 使用 DeepSeek 驱动的 Agent 基于资料生成回答；Agent 可搜索当前知识库、阅读已检索到的原文，或在缺少必要信息时向用户提问。
- 通过网页管理账号、对话和工作区；工作区权限由后端校验。
- 通过服务器发送事件（SSE）流式输出回答，并在界面展示资料来源。

## 工作流程

```text
文档与图片
  → 解析、OCR 和视觉分析
  → 置信度检查与分层切片
  → BM25 + 多模态稠密检索
  → 倒数排名融合与 MMR 去重
  → 重排
  → Agent 按需搜索、阅读原文或追问
  → 来源与回答校验
  → FastAPI / SSE → React
```

Agent 当前可调用以下三种工具：

| 工具 | 范围 |
| --- | --- |
| `search_knowledge(query)` | 在当前授权工作区内检索，并返回排序后的证据片段。 |
| `read_source(evidence_id)` | 读取本次请求检索到的证据原文；不能打开任意文件路径。 |
| `collect_user_info(question)` | 仅在回答依赖用户才能提供的信息时提出一个问题。 |

## 环境要求

- Python 3.11
- Node.js 和 npm
- 可访问的 Milvus 实例（默认向量存储配置需要；参阅 [部署方式](https://milvus.io/docs/install-overview.md)）
- 已配置的语言模型和多模态模型服务凭据

OCR 和文档解析依赖较大，安装可能需要一些时间。模型 API 调用可能产生服务商费用。

## 快速开始

### 1. 配置后端

```bash
git clone https://github.com/zhe-w12/Multimodal-knowledge-base.git
cd Multimodal-knowledge-base
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/install_nltk_data.py
cp .env.example .env
```

Windows PowerShell 用户请使用 `py -3.11 -m venv .venv` 创建环境，运行 `.\.venv\Scripts\Activate.ps1` 激活环境，并使用 `Copy-Item .env.example .env` 创建本地配置。之后运行相同的 `python -m pip install -r requirements.txt` 和 `python scripts/install_nltk_data.py` 命令。

编辑 `.env`，填写 `DEEPSEEK_API_KEY`、`BAILIAN_API_KEY` 和 `BAILIAN_WORKSPACE_ID`。启动 Milvus，并设置 `MILVUS_ENDPOINT`；如果你的部署需要令牌，也要设置 `MILVUS_TOKEN`。后端会从仓库根目录读取 `.env`。Redis 和 MySQL 集成是可选项，默认关闭；账号数据默认使用本地 SQLite，除非配置了 `AUTH_DATABASE_URL`。

| 变量 | 是否必需 | 用途 |
| --- | --- | --- |
| `DEEPSEEK_API_KEY` | 是 | 主聊天模型凭据。 |
| `BAILIAN_API_KEY` | 默认模型服务必需 | 向量化、重排和视觉分析服务凭据。 |
| `BAILIAN_WORKSPACE_ID` | 未设置 `BAILIAN_API_HOST` 时必需 | 用于构造默认百炼服务端点。 |
| `MILVUS_ENDPOINT` | 默认向量存储必需 | Milvus 服务地址。 |
| `MILVUS_TOKEN` | 视部署而定 | Milvus 启用身份验证时使用的令牌。 |
| `RAG_CORS_ORIGINS` | 建议设置 | 允许访问 API 的前端来源，多个来源以逗号分隔。 |
| `RAG_KB_RUNTIME_PATH` | 否 | 知识库数据的本地持久化路径。 |
| `AUTH_DATABASE_URL` | 否 | 账号数据库地址；默认使用本地 SQLite。 |

不要提交 `.env`、API Key、密码、上传资料或本地数据库。服务商凭据应保存在后端，不要放入前端 `VITE_*` 变量。

### 2. 启动 API

```bash
python -m uvicorn manufacturing_rag.api:app --host 127.0.0.1 --port 8000 --reload
```

API 地址为 `http://127.0.0.1:8000`。FastAPI 交互式接口文档位于 `http://127.0.0.1:8000/docs`，健康检查接口为 `http://127.0.0.1:8000/health`。

### 3. 启动前端

在第二个终端中运行：

```bash
cd frontend
npm ci
npm run dev
```

打开 `http://127.0.0.1:5173`。如果后端运行在其他主机或端口，请在前端环境中设置 `VITE_API_BASE_URL`，并在后端 `RAG_CORS_ORIGINS` 配置中允许该前端来源。 回答生成期间可以继续浏览文档、切换或新建对话，生成会在后台持续进行。同一对话中追加的问题会排队等待当前回答完成，也可以随时停止当前回答。

## 配置与数据

`.env.example` 列出了主要连接配置。模型名称、检索参数和可选服务可通过环境变量调整；支持的配置项和默认值见 [`manufacturing_rag/config.py`](manufacturing_rag/config.py)。

默认情况下，知识库上传数据保存在 `tmp/`，账号数据和自动生成的认证密钥保存在 `data/auth/`。这些是本地运行数据，已从 Git 中排除，应单独备份并妥善保护。用于向量化、重排、视觉分析或生成回答的文档内容可能会由已配置的外部服务商处理；上传敏感资料前，请先查看其条款和数据政策。

## API 概览

主要接口包括 `GET /health`、`POST /answer`、`POST /answer/stream` 和 `POST /kb/document`。完整且最新的接口定义可在运行中的服务 `/docs` 页面查看。

## 开发

运行后端测试并构建前端：

```bash
pytest -q
cd frontend
npm ci
npm run build
```

贡献方式见 [CONTRIBUTING.md](CONTRIBUTING.md)；私下报告安全漏洞请参阅 [SECURITY.md](SECURITY.md)。

## 安全与隐私

在本地开发环境以外部署时，请使用 HTTPS 并明确配置允许访问的来源。请通过适当的访问控制和备份保护上传文件、账号数据及自动生成的密钥。安全漏洞报告方式见 [SECURITY.md](SECURITY.md)。

## 使用限制

对于低分辨率、遮挡或复杂图表，OCR 和模型生成的描述可能不准确。请对照原始资料核实提取内容。回答质量取决于上传资料、模型服务和检索配置。

## 许可证

本项目采用 [MIT 许可证](LICENSE)。第三方依赖、模型服务、模型权重和用户提供的资料仍受其各自许可证和服务条款约束。
