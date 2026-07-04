# KAAS-RAG

基于 LangGraph 的 RAG（检索增强生成）知识库系统，面向产品手册的智能问答平台。

## 核心功能

- **知识库导入**：PDF（MinerU API）/ Markdown 文件 → 自动分块 → BGE-M3 向量化 → Milvus 入库
- **智能问答**：多轮对话 + 商品名确认 + 4 路并行检索 → RRF 融合 → Reranker 精排 → LLM 生成
- **知识库管理**：多知识库 CRUD，30+ 可调参数 JSON 配置，API 动态管理
- **检索评估**：数据集管理 + 多引擎评估（自定义 + RAGAS）+ 参数横向对比

## 快速开始

```bash
# 1. 安装依赖
uv sync

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env，填入 API Key 和服务地址

# 3. 启动外部服务（Milvus / MySQL / MongoDB / MinIO）
docker compose -f docker/docker-compose.yml up -d

# 4. 启动应用
python app/main.py
# 访问 http://localhost:8888
```

## 架构概览

```
用户 → FastAPI (:8888) → LangGraph 工作流
                          ├── 导入流程：PDF/MD → 分块 → 向量化 → Milvus
                          └── 查询流程：意图确认 → 4路并行检索 → RRF → Rerank → 生成
                              ├── 向量搜索 (BGE-M3 稠密+稀疏)
                              ├── HyDE 假设文档搜索
                              ├── 知识图谱查询 (Neo4j)
                              └── 网络搜索 (DashScope MCP)

依赖服务：Milvus · MySQL · MongoDB · MinIO · Neo4j(可选) · BGE-Reranker
```

## 技术栈

| 类别 | 技术 |
|---|---|
| Web 框架 | FastAPI + SSE 流式 |
| 工作流 | LangGraph (Send API 并行分叉) |
| LLM | DeepSeek / DashScope (OpenAI 兼容) |
| Embedding | BGE-M3 1024D 稠密 + 稀疏向量 |
| 向量库 | Milvus 混合搜索 (COSINE + IP) |
| 重排序 | BGE-Reranker-large |
| 存储 | MySQL + MongoDB + MinIO + Neo4j |

## 项目结构

```
kaas-rag/
├── app/
│   ├── main.py                 # FastAPI 入口
│   ├── api/                    # API 层（路由 + 服务）
│   ├── import_process/agent/   # 导入工作流（7 节点）
│   ├── query_process/agent/    # 查询工作流（9 节点）
│   ├── conf/                   # 配置管理
│   ├── clients/                # 外部服务客户端
│   ├── lm/                     # LLM / Embedding / Reranker
│   ├── core/                   # 日志 + Prompt 加载
│   ├── utils/                  # 工具函数
│   ├── eval/                   # 检索评估模块
│   └── page/                   # 前端页面
├── prompts/                    # Prompt 模板
├── docs/                       # 项目文档
├── docker/                     # Docker Compose
├── test/                       # 测试文件
└── sql/                        # 数据库初始化 SQL
```

## 文档

| 文档 | 说明 |
|---|---|
| [架构设计文档](docs/架构设计文档.md) | 系统架构、工作流详解、API 接口、目录结构 |
| [技术问题与解决方案](docs/技术问题与解决方案.md) | 5 个深度技术问题的分析与解决方案 |
| [技术亮点](docs/技术亮点.md) | 12 大技术亮点详解 |
| [项目说明](docs/项目说明.md) | 项目定位、生产化注意事项、部署架构 |
| [数据库与部署说明](docs/数据库与部署说明.md) | 表结构、DDL、初始化流程、部署步骤 |
| [系统可调参数梳理](docs/系统可调参数梳理.md) | 30+ 参数的详细说明与调优建议 |

## 注意事项

当前工程已完成 RAG 核心模块用于技术沉淀和验证学习，满足生产级高精度检索需求。如需投入生产环境，需补充：

- 接口鉴权（API Key / JWT / OAuth2）
- 数据越权校验（用户 → 知识库权限映射）
- 性能优化（Redis 任务状态、Embedding 水平扩展、LLM 限流熔断）
- 安全加固（密钥管理、文件上传白名单）
- 监控告警（Prometheus + Grafana）
