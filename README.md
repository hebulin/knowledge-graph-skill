# KG Skill - 知识图谱技能

一个自包含的知识图谱引擎，用于构建、查询和推理实体-关系图谱。专为 **GraphRAG**、**Agent 长期记忆** 和 **企业知识资产管理** 设计。

## 为什么需要 KG Skill？

传统向量 RAG 在多跳推理和全局视野上存在瓶颈。KG Skill 通过从文档中构建结构化知识图谱，融合向量检索、图遍历和社区级摘要来进行检索，从根本上解决这个问题。

| 能力 | 传统向量 RAG | KG Skill |
|------|-------------|----------|
| 多跳推理 | 弱（依赖 LLM 隐式推理） | 强（显式图遍历 + 路径推理） |
| 全局视野 | 局部 Top-K 相似 | 社区发现 + 全局摘要 |
| 知识溯源 | 仅文本引用 | 实体 -> Chunk -> 原文三级溯源 |
| 幻觉控制 | 依赖 Prompt 工程 | 结构化约束 + 置信度过滤 |

## 快速开始

### 作为 Codex 技能安装

```bash
# 在另一个 Codex 实例中通过 skill-installer 安装
python scripts/install-skill-from-github.py --repo <你的用户名>/<本仓库名> --path knowledge-graph-skill
```

安装后技能自动出现在 Codex 的 Skills 列表中，Agent 遇到知识图谱相关任务时自动触发。

### 启动 API 服务

```bash
cd knowledge-graph-skill
pip install -r assets/requirements.txt
export OPENAI_API_KEY=sk-...     # 可选，启用 LLM 抽取和 Text2Cypher
python scripts/kg_server.py --port 8700
```

服务启动后访问 `http://localhost:8700`：
- 14 个 REST API 接口（`/docs` 查看 Swagger 文档）
- OpenAI 兼容的 Tool Calling 定义（`GET /api/v1/tools`）
- 首次运行自动创建 SQLite 数据库 `~/.knowledge-graph-skill/kg.db`，无需外部数据库

### 作为 Python 库使用

```python
import sys; sys.path.insert(0, "knowledge-graph-skill/scripts")
from kg_core import KGStore
from kg_extract import ExtractionPipeline

store = KGStore()
pipeline = ExtractionPipeline(store)

# 从文本构建图谱
pipeline.extract("苹果公司由史蒂夫·乔布斯于1976年创立。蒂姆·库克是现任CEO。")

# 查询
results = store.search_entities("苹果")
subgraph = store.query_subgraph(entity_id=results[0]["entity_id"], depth=2)
```

### 配合 LLM Tool Calling 使用

```python
import requests

# 获取工具定义
tools = requests.get("http://localhost:8700/api/v1/tools").json()

# 注册给 LLM
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "苹果公司收购了哪些公司？"}],
    tools=tools,
)
```

## 目录结构

```
knowledge-graph-skill/
├── SKILL.md                     # Agent 入口（触发条件 + 使用指令）
├── agents/openai.yaml           # UI 元数据
├── scripts/
│   ├── kg_core.py              # 数据模型、存储、CRUD、实体消歧
│   ├── kg_extract.py           # 混合抽取（规则 + LLM）
│   ├── kg_graphrag.py          # GraphRAG 检索、Text2Cypher、社区发现
│   ├── kg_server.py            # FastAPI 服务（14 个接口 + Tool Calling）
│   └── kg_export.py            # Mermaid / JSON-LD / 文本树 / CSV / GraphML 导出
├── references/                  # 详细文档（按需加载）
├── assets/
│   ├── schemas/                # JSON Schema（实体 / 关系 / 事件）
│   ├── config/                 # default_config.yaml 配置文件
│   ├── docker/                 # Dockerfile + docker-compose.yml
│   └── requirements.txt
└── license.txt
```

## 存储模式

| 模式 | 图存储 | 向量存储 | 外部依赖 | 适用规模 |
|------|--------|---------|---------|---------|
| **轻量模式**（默认） | SQLite + NetworkX | NumPy | 无 | <5万节点 |
| **生产模式** | Neo4j 5.x | Qdrant | Docker | >10万节点 |

在 `assets/config/default_config.yaml` 中切换：

```yaml
storage:
  mode: production               # lightweight | production
```

### 生产环境部署

```bash
cd knowledge-graph-skill/assets/docker
docker-compose up -d             # 启动 API + Neo4j + Qdrant
```

## API 接口列表

| 序号 | 方法 | 路径 | 功能说明 |
|------|------|------|---------|
| 1 | POST | `/api/v1/extract` | 从文档抽取知识并入库 |
| 2 | POST | `/api/v1/entities` | 创建实体（自动消歧） |
| 3 | PATCH | `/api/v1/entities/{id}` | 更新实体属性 |
| 4 | DELETE | `/api/v1/entities/{id}` | 软删除实体 |
| 5 | POST | `/api/v1/relations` | 创建关系（校验约束） |
| 6 | POST | `/api/v1/search/entities` | 混合实体搜索 |
| 7 | POST | `/api/v1/graph/subgraph` | N跳子图抽取 |
| 8 | POST | `/api/v1/query/text2cypher` | 自然语言转图查询 |
| 9 | POST | `/api/v1/graphrag/search` | GraphRAG 混合检索 |
| 10 | POST | `/api/v1/graph/paths` | 实体间路径查找 |
| 11 | POST | `/api/v1/reason` | 神经-符号混合推理 |
| 12 | GET | `/api/v1/stats` | 图谱统计信息 |
| 13 | POST | `/api/v1/export` | 导出为多种格式 |
| 14 | POST | `/api/v1/import` | 批量导入 |

完整接口文档见 `knowledge-graph-skill/references/api_reference.md`。

## 核心特性

- **混合抽取**：规则 NER（快速、低成本）+ LLM 辅助抽取（高精度），自动按内容复杂度分流
- **四阶段实体消歧**：精确匹配 -> 别名匹配 -> 语义相似度 -> LLM 判定
- **原文溯源**：每条三元组绑定到原文 Chunk，支持"知识 -> Chunk -> 原文"三级回溯
- **GraphRAG 混合检索**：向量 + 图遍历 + 社区摘要，通过 RRF 融合排序
- **Text2Cypher**：自然语言转图查询，含 AST 校验和自动修复
- **神经-符号推理**：规则推理 + LLM 推理，输出可解释的推理链
- **时序图谱**：支持时间维度的属性和关系查询
- **软删除 + 生命周期**：实体经历 active -> decaying -> deprecated -> archived 的完整生命周期

## 配置说明

详见 `knowledge-graph-skill/assets/config/default_config.yaml`，包含 LLM 模型、Embedding 维度、抽取策略、GraphRAG 参数和安全设置等全部配置项。

## 许可证

MIT 许可证，详见 `knowledge-graph-skill/license.txt`。

## 设计文档

完整的架构与设计方案文档见 `知识图谱技能架构设计方案.md`（中文），涵盖八大章节：技能定位、治理规范、核心功能模块、技术架构、数据模型、接口设计、端到端使用流程、非功能性需求。
