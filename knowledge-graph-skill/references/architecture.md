# Architecture Overview

## Table of Contents
- [System Layers](#system-layers)
- [Storage Modes](#storage-modes)
- [Tech Stack](#tech-stack)
- [Deployment](#deployment)
- [Performance](#performance)

## System Layers

| Layer | Responsibility | Components |
|-------|---------------|------------|
| Access | API gateway, Tool Calling | FastAPI server, OpenAPI docs |
| Orchestration | Auth, rate limiting, cost control | (pluggable) |
| Compute | Extraction, reasoning, retrieval | kg_extract, kg_graphrag, kg_query |
| Storage | Graph + vector + relational | SQLite+NetworkX (lightweight) / Neo4j+Qdrant (production) |

## Storage Modes

### Lightweight (default)
- **Graph**: SQLite (metadata) + NetworkX (topology, in-memory)
- **Vector**: NumPy cosine similarity (in-memory)
- **No external dependencies** beyond pip packages
- Suitable for: dev, prototype, <50K nodes

### Production
- **Graph**: Neo4j 5.x (causal cluster)
- **Vector**: Qdrant (HNSW index)
- **Relational**: PostgreSQL (audit logs, document metadata)
- **Cache**: Redis
- Suitable for: enterprise, >100K nodes

Switch via `assets/config/default_config.yaml` -> `storage.mode: production`

## Tech Stack

| Component | Default | Production Alternative |
|-----------|---------|----------------------|
| Language | Python 3.11+ | - |
| Graph DB | SQLite + NetworkX | Neo4j / NebulaGraph |
| Vector DB | NumPy (in-memory) | Qdrant / Milvus |
| LLM | OpenAI gpt-4o | Azure OpenAI / vLLM |
| Embedding | text-embedding-3-small | bge-m3 (multilingual) |
| API | FastAPI + Uvicorn | - |
| Container | Docker | Kubernetes |

## Deployment

### Docker (recommended for production)

```bash
cd assets/docker
docker-compose up -d
```

Services started:
- `kg-api` (port 8700)
- `neo4j` (port 7687/7474)
- `qdrant` (port 6333)

### Local development

```bash
pip install -r assets/requirements.txt
python scripts/kg_server.py --port 8700
```

## Performance (lightweight mode, 10K nodes)

| Operation | P50 | P95 |
|-----------|-----|-----|
| Entity search (keyword) | 5ms | 20ms |
| Subgraph (2-hop, 50 nodes) | 10ms | 40ms |
| Path finding (3-hop) | 15ms | 60ms |
| GraphRAG hybrid | 50ms | 200ms |
| Extraction (per chunk, LLM) | 1s | 3s |

Production mode with Neo4j+Qdrant targets <200ms P95 for all query types at 100K nodes.
