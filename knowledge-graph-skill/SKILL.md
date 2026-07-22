---
name: knowledge-graph-skill
description: >-
  Build, manage, and query knowledge graphs for GraphRAG, entity-relationship
  extraction, and multi-hop reasoning. Use this skill when: (1) extracting
  entities, relations, and events from documents (text, markdown, JSON, tables)
  into a structured knowledge graph, (2) performing GraphRAG hybrid retrieval
  that combines vector search, graph traversal, and community summaries for
  answering complex multi-hop questions, (3) querying the graph via natural
  language (Text2Cypher), subgraph extraction, or path finding between entities,
  (4) managing graph lifecycle including entity CRUD, deduplication, temporal
  versioning, and conflict resolution, (5) exporting graphs to Mermaid,
  Graphviz, JSON-LD, or text-tree formats, (6) running neuro-symbolic reasoning
  over the graph for inference and root-cause analysis. Also use when an AI
  Agent needs a persistent, queryable knowledge base with provenance tracking
  and confidence scoring as an alternative or complement to vector-only RAG.
---

# Knowledge Graph Skill (KG Skill)

A self-contained knowledge graph engine for building, querying, and reasoning
over entity-relationship graphs. Designed for GraphRAG, Agent memory, and
enterprise knowledge management.

## Quick Start

### 1. Install dependencies

```bash
pip install -r assets/requirements.txt
```

### 2. Start the API server (all 14 endpoints)

```bash
python scripts/kg_server.py --host 0.0.0.0 --port 8700
```

The server auto-creates a SQLite database at `~/.knowledge-graph-skill/kg.db` on first run.
No external databases required for the default lightweight mode.

### 3. Use as a Python library

```python
from kg_core import KGStore

store = KGStore()                          # opens SQLite + in-memory graph
store.extract_from_text("Apple acquired Beats for $3B in 2014.")
results = store.search("AI companies")
subgraph = store.query_subgraph(entity_id, depth=2)
```

### 4. Use as LLM Tool Calling

The server exposes OpenAI-compatible function-calling schemas at
`GET /api/v1/tools`. Register these as tools for any LLM agent. See
[references/tool_definitions.md](references/tool_definitions.md) for all 14
tool definitions.

## Architecture Overview

```
knowledge-graph-skill/
├── SKILL.md                      # This file
├── agents/openai.yaml            # UI metadata
├── scripts/
│   ├── kg_core.py               # Core: data model, storage, CRUD, resolution
│   ├── kg_extract.py            # Extraction: document -> entities/relations
│   ├── kg_query.py              # Query: subgraph, paths, neighbors, search
│   ├── kg_graphrag.py           # GraphRAG: vector + graph + community fusion
│   ├── kg_server.py             # FastAPI server: 14 REST + Tool Calling APIs
│   ├── kg_export.py             # Export: Mermaid, JSON-LD, text-tree, CSV
│   ├── kg_pii.py                 # PII detection & masking
│   ├── kg_sync.py                # Document sync (file watcher + git diff)
│   └── test_basic.py             # Smoke tests
├── references/
│   ├── api_reference.md         # All 14 API endpoints documented
│   ├── data_model.md            # Entity/Relation/Event JSON Schemas
│   ├── tool_definitions.md      # LLM Function Calling definitions
│   └── architecture.md          # Full architecture and deployment guide
├── assets/
│   ├── schemas/                 # JSON Schema files (entity/relation/event)
│   ├── config/default_config.yaml
│   ├── docker/                  # Docker Compose + Dockerfile
│   └── requirements.txt
└── license.txt
```

## Storage Modes

| Mode | Graph Store | Vector Store | When to Use |
|------|------------|-------------|-------------|
| **Lightweight** (default) | SQLite + NetworkX (in-memory) | NumPy cosine similarity | Dev, prototype, <50K nodes |
| **Production** | Neo4j 5.x | Qdrant | Enterprise, >100K nodes |

Switch by editing `assets/config/default_config.yaml`. The lightweight mode
works out-of-the-box with zero external dependencies beyond pip packages.

## Core Workflows

### Workflow 1: Build a Knowledge Graph from Documents

1. Call `extract_knowledge` with document content (text/markdown/JSON/table).
2. The extraction pipeline chunks the document, runs rule-based NER (fast) then
   LLM-assisted extraction (for complex relations), and generates triples with
   confidence scores.
3. Entities are auto-resolved against existing graph (4-stage cascade: exact ->
   alias -> semantic similarity -> LLM judgment).
4. Each triple is bound to its source chunk (Source Grounding) for provenance.
5. Results are persisted to the graph store and vector index.

```python
# Via API
POST /api/v1/extract
{"content": "Apple was founded by Steve Jobs in 1976...", "format": "text"}

# Via Python
store.extract_from_text(content, format="text", auto_resolve=True)
```

### Workflow 2: GraphRAG Hybrid Retrieval

1. Call `graphrag_search` with a natural-language query.
2. The system decomposes the query and runs parallel retrieval:
   - Vector search (semantic similarity on entity descriptions)
   - Graph traversal (multi-hop from seed entities)
   - Community summaries (global context via Leiden algorithm)
3. Results are fused via Reciprocal Rank Fusion (RRF).
4. Returns structured context (entities + relations + summaries + source chunks)
   ready for LLM answer generation.

```python
# Via API
POST /api/v1/graphrag/search
{"query": "Apple's AI patent strategy vs competitors", "strategy": "auto"}

# Via Python
context = store.graphrag_search("Apple's AI patent strategy", strategy="auto")
```

### Workflow 3: Natural Language Graph Query (Text2Cypher)

1. Call `text2cypher` with a natural-language question.
2. The system injects graph schema + few-shot examples into an LLM prompt.
3. Generated Cypher is validated (AST check) and auto-repaired if invalid (max 2
   retries).
4. Validated query executes in read-only mode; results returned as structured
   JSON.

```python
POST /api/v1/query/text2cypher
{"question": "Which companies did Apple acquire after 2020?"}
```

### Workflow 4: Agent Multi-Step Exploration

An AI Agent can chain multiple tool calls for complex tasks:

1. `search_entities("payment service")` -> find the entity
2. `query_subgraph(entity_id, depth=3)` -> explore dependencies
3. `text2cypher("recent changes to dependencies")` -> find changes
4. `reason("does this change cause the latency?")` -> infer root cause

The tool descriptions in `references/tool_definitions.md` are written so LLMs
can autonomously select and sequence calls.

## Configuration

Edit `assets/config/default_config.yaml`:

```yaml
storage:
  mode: lightweight          # lightweight | production
  sqlite_path: "~/.knowledge-graph-skill/kg.db"
  neo4j_uri: "bolt://localhost:7687"    # production mode only
  qdrant_url: "http://localhost:6333"   # production mode only

extraction:
  strategy: auto             # auto | rule_first | llm_first
  chunk_size: 512
  chunk_overlap: 64
  auto_resolve: true

llm:
  model: gpt-4o
  api_base: ""               # empty = use OPENAI_API_KEY env
  max_tokens: 4096
  temperature: 0.1

embedding:
  model: text-embedding-3-small
  dimension: 1536

graphrag:
  default_strategy: auto     # auto | vector | graph | hybrid | community
  max_context_entities: 20
  min_confidence: 0.6
  community_algorithm: leiden

security:
  pii_detection: true
  pii_masking: true
  read_only_text2cypher: true
```

## Key Design Decisions

- **Property Graph model**: chosen over hypergraph for mainstream DB support
  and Text2Cypher effectiveness. N-ary relations expressed via event nodes.
- **Hybrid extraction**: rules first (cheap), LLM second (accurate). Typical
  cost split: 30% direct mapping, 40% rule+small-model, 25% LLM, 5% manual.
- **4-stage entity resolution**: exact -> alias -> semantic (cosine > 0.85) ->
  LLM judgment. Low-confidence matches go to review queue.
- **Source Grounding**: every triple binds to its source chunk for provenance.
  Answers cite original text, reducing hallucination.
- **Soft delete + temporal versioning**: deleted entities are marked, not
  removed. Attribute history is preserved for time-travel queries.

## Security Features

- **PII Detection & Masking**: Automatically detects and masks emails, phone
  numbers, ID cards, bank cards, and IP addresses before extraction. Configure
  via security.pii_detection in config. See scripts/kg_pii.py.
- **API Key Authentication**: Set KG_API_KEY environment variable to require
  X-API-Key header on all business endpoints. Health check and tool
  definitions remain public. Disabled by default for development.

## Document Sync

Keep the knowledge graph in sync with source files:

`ash
# File watcher mode (monitors a directory)
python scripts/kg_sync.py watch --path /path/to/docs --port 8700

# Git diff mode (process changed files from last commit)
python scripts/kg_sync.py git-diff --repo /path/to/repo --ref HEAD~1
`

- Modified files are re-extracted and merged into the graph
- Deleted files trigger deprecation of associated knowledge entities
- Requires watchdog package for file watcher mode

## Testing

Run basic smoke tests:

`ash
python scripts/test_basic.py
`

Tests cover: entity CRUD, deduplication, relation queries, PII detection,
extraction pipeline, graph export, and statistics.

## Production Deployment

For enterprise scale (>100K nodes), use Docker Compose:

```bash
cd assets/docker
docker-compose up -d   # starts Neo4j + Qdrant + PostgreSQL + Redis + API
```

See [references/architecture.md](references/architecture.md) for full
deployment guide, K8s manifests, and performance benchmarks.

## References

- **[api_reference.md](references/api_reference.md)**: All 14 API endpoints with
  parameters, return values, and examples.
- **[data_model.md](references/data_model.md)**: Entity, Relation, and Event
  JSON Schemas with field descriptions.
- **[tool_definitions.md](references/tool_definitions.md)**: OpenAI Function
  Calling definitions for all 14 tools (copy-paste ready for agent setup).
- **[architecture.md](references/architecture.md)**: Full architecture, tech
  stack selection, deployment modes, and non-functional requirements.
