# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

kaas-rag 是一个基于 LangGraph 的 RAG（检索增强生成）知识库系统，用于产品手册的智能问答。核心流程分为两路：**知识库导入**和**查询检索**。

## 常用命令

### 启动服务

```bash
# 统一服务入口（端口 8000，包含文件导入 + 智能问答）
python app/main.py
```

### 运行测试

```bash
# 测试文件位于 test/ 目录，直接运行即可
python test/04-test_graph_flow.py
python test/05-test-main-graph.py
```

### 依赖管理

使用 uv 管理依赖，`uv.lock` 和 `pyproject.toml` 定义所有依赖。Python 版本要求 >= 3.11。

## 架构

### 两层 LangGraph 工作流

**1. 知识库导入流程** (`app/import_process/agent/main_graph.py`)

```
入口(node_entry) → [条件分支]
  ├── PDF路径 → PDF转MD(node_pdf_to_md) → MD图片处理(node_md_img)
  └── MD直接导入 → MD图片处理(node_md_img)
→ 文档分块(node_document_split) → 主体名称识别(node_item_name_recognition)
→ BGE向量化(node_bge_embedding) → 导入Milvus(node_import_milvus)
```

状态定义在 `ImportGraphState`（`app/import_process/agent/state.py`），控制标记 `is_pdf_read_enabled` / `is_md_read_enabled` 决定分支走向。每个节点函数接收 state 并返回字典更新状态。

**2. 查询检索流程** (`app/query_process/agent/main_graph.py`)

```
意图确认(node_item_name_confirm) → [条件分支]
  ├── 已有answer（反问/拒答）→ 直接输出(node_answer_output)
  └── 继续检索 → 多路并行搜索(node_multi_search):
       ├── 向量搜索(node_search_embedding)
       ├── HyDE搜索(node_search_embedding_hyde)
       ├── 知识图谱查询(node_query_kg)
       └── 网络搜索MCP(node_web_search_mcp)
     → 结果合并(node_join) → RRF融合(node_rrf) → 重排序(node_rerank) → 答案生成(node_answer_output)
```

`node_multi_search` 和 `node_join` 是纯虚拟节点（`lambda x: x`），仅用于 LangGraph 的分叉/合并中转。

### 目录结构

| 目录 | 用途 |
|---|---|
| `app/api/` | 统一 API 层：路由定义(`routes/`)、后台服务(`services/`)、Pydantic 模型(`models.py`) |
| `app/main.py` | FastAPI 统一入口，注册中间件和所有路由，端口 8000 |
| `app/conf/` | 配置类（dataclass），从 `.env` 读取，包括 LLM、Milvus、MinIO、Embedding、Reranker |
| `app/clients/` | 外部服务客户端单例：Milvus、MinIO、MongoDB、Neo4j |
| `app/core/` | 日志（基于 loguru）、Prompt 模板加载（支持 `{变量}` 占位符渲染） |
| `app/lm/` | LLM 客户端（OpenAI 兼容 API，带缓存）、Embedding 工具、Reranker 工具 |
| `app/utils/` | 通用工具：SSE 推送、任务状态追踪、路径工具、字符串转义等 |
| `app/import_process/agent/nodes/` | 导入流程各节点实现 |
| `app/query_process/agent/nodes/` | 查询流程各节点实现 |
| `prompts/` | Prompt 模板文件（`.prompt` 后缀），通过 `load_prompt(name, **kwargs)` 加载 |

### 关键技术细节

- **LLM**：通过 LangChain 的 `ChatOpenAI` 兼容阿里百炼 DashScope API（`OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1`），支持千问系列模型。客户端有全局缓存，键为 `(模型名, JSON模式)`。
- **Embedding**：使用 BGE-M3 模型，生成稠密向量（1024维）+ 稀疏向量，存入 Milvus 做混合搜索。模型从 ModelScope 下载到本地。
- **Milvus 混合搜索**：稠密向量用 COSINE 相似度，稀疏向量用 IP（内积），通过 `WeightedRanker` 加权融合（默认权重 0.5:0.5）。
- **PDF 解析**：使用 MinerU API（`mineru.net/api/v4`）将 PDF 转为 Markdown。
- **任务追踪**：`app/utils/task_utils.py` 使用内存字典追踪任务状态（pending → processing → completed/failed），支持前端轮询和 SSE 推送。
- **配置优先级**：`.env` 文件 → `app/conf/*.py` 中的 dataclass。所有环境变量通过 `load_dotenv()` 加载。
- **统一 FastAPI 服务**：单服务端口 8000，所有路由扁平化注册，前端为统一页面（`app/page/index.html`）含两个 Tab（知识库导入 / 智能问答）。API 层分层为 `routes/`（路由定义）、`services/`（后台任务）。

### 外部服务依赖

启动前需要确保以下服务可用（配置在 `.env` 中）：
- **Milvus**：向量数据库（`MILVUS_URL`）
- **MinIO**：对象存储（`MINIO_ENDPOINT`）
- **MongoDB**：对话历史（`MONGO_URL`）
- **Neo4j**：知识图谱（`NEO4J_URI`，可选）
- **MinerU API**：PDF 解析（`MINERU_BASE_URL`）
- **DashScope API**：LLM 调用和 MCP 网络搜索
