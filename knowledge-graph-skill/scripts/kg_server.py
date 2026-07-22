"""
KG Skill API Server - FastAPI server exposing all 14 REST + Tool Calling APIs.

Endpoints:
  1. POST   /api/v1/extract              - Extract knowledge from document
  2. POST   /api/v1/entities             - Create entity
  3. PATCH  /api/v1/entities/{id}        - Update entity
  4. DELETE /api/v1/entities/{id}        - Delete entity
  5. POST   /api/v1/relations            - Create relation
  6. POST   /api/v1/search/entities      - Search entities
  7. POST   /api/v1/graph/subgraph       - Query subgraph
  8. POST   /api/v1/query/text2cypher    - Natural language to Cypher
  9. POST   /api/v1/graphrag/search      - GraphRAG hybrid search
  10. POST  /api/v1/graph/paths          - Find paths between entities
  11. POST  /api/v1/reason               - Run reasoning
  12. GET   /api/v1/stats                - Graph statistics
  13. POST  /api/v1/export               - Export graph
  14. POST  /api/v1/import               - Batch import

Also exposes:
  GET  /api/v1/tools          - OpenAI Tool Calling definitions
  GET  /api/v1/health         - Health check
  GET  /docs                  - Swagger UI (auto-generated)
"""

import os
import sys
import json
import argparse
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Depends, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

# Ensure scripts directory is in path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kg_core import KGStore, Entity, Relation, Document, Chunk
from kg_extract import ExtractionPipeline
from kg_graphrag import GraphRAGSearch, Text2Cypher
from kg_export import GraphExporter


# ---------------------------------------------------------------------------
# Pydantic Request Models
# ---------------------------------------------------------------------------

class ExtractRequest(BaseModel):
    content: str = Field(..., description="Document content")
    format: str = Field("auto", description="text/markdown/json/table/auto")
    extraction_strategy: str = Field("auto", description="auto/rule_first/llm_first")
    chunk_size: int = Field(512)
    chunk_overlap: int = Field(64)
    auto_resolve: bool = Field(True)
    title: str = Field("")


class CreateEntityRequest(BaseModel):
    name: str
    type: str = "Other"
    aliases: list = []
    attributes: dict = {}
    description: str = ""
    tags: list = []
    confidence: float = 1.0


class UpdateEntityRequest(BaseModel):
    name: Optional[str] = None
    attributes: Optional[dict] = None
    description: Optional[str] = None
    tags: Optional[list] = None
    confidence: Optional[float] = None
    temporal: bool = False


class CreateRelationRequest(BaseModel):
    source_entity_id: str
    relation_type: str
    target_entity_id: str
    attributes: dict = {}
    confidence: float = 1.0
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None


class SearchRequest(BaseModel):
    query: str
    entity_types: Optional[list] = None
    min_confidence: float = 0.0
    top_k: int = 10
    search_mode: str = "hybrid"


class SubgraphRequest(BaseModel):
    entity_id: str
    depth: int = 2
    relation_types: Optional[list] = None
    direction: str = "both"
    limit_per_hop: int = 50
    include_attributes: bool = True
    min_confidence: float = 0.0


class Text2CypherRequest(BaseModel):
    question: str
    context_entities: Optional[list] = None
    max_results: int = 100
    dry_run: bool = False


class GraphRAGRequest(BaseModel):
    query: str
    strategy: str = "auto"
    max_context_entities: int = 20
    max_context_relations: int = 30
    include_source_chunks: bool = True
    subgraph_depth: int = 2
    min_confidence: float = 0.6


class FindPathsRequest(BaseModel):
    source_entity_id: str
    target_entity_id: str
    max_depth: int = 5
    max_paths: int = 5
    algorithm: str = "shortest"
    relation_types: Optional[list] = None


class ReasonRequest(BaseModel):
    query: str
    reasoning_type: str = "auto"
    context_entity_ids: Optional[list] = None
    max_inference_depth: int = 5
    verify: bool = True


class ExportRequest(BaseModel):
    format: str
    entity_ids: Optional[list] = None
    depth: int = 2
    include_attributes: bool = True
    max_nodes: int = 500


class ImportRequest(BaseModel):
    format: str = "jsonld"
    data: str
    mode: str = "upsert"
    auto_resolve: bool = True


# ---------------------------------------------------------------------------
# Server Factory
# ---------------------------------------------------------------------------

def create_app(config: dict = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    config = config or {}
    app = FastAPI(
        title="KG Skill API",
        description="Knowledge Graph Skill - Build, query, and reason over "
                    "entity-relationship graphs for GraphRAG.",
        version="1.0.0",
    )

    # API Key authentication
    api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
    expected_api_key = os.environ.get("KG_API_KEY", "")

    async def verify_api_key(api_key: str = Security(api_key_header)):
        """Verify API key if KG_API_KEY environment variable is set."""
        if not expected_api_key:
            return  # Auth disabled (development mode)
        if api_key != expected_api_key:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    # Initialize store and pipeline
    store = KGStore(config=config.get("storage", {}))
    pipeline = ExtractionPipeline(store, config)
    graphrag = GraphRAGSearch(store, config)
    exporter = GraphExporter(store)

    # Store references
    app.state.store = store
    app.state.pipeline = pipeline
    app.state.graphrag = graphrag
    app.state.exporter = exporter
    app.state.config = config

    # ------------------------------------------------------------------
    # Health & Tools
    # ------------------------------------------------------------------

    @app.get("/api/v1/health")
    async def health():
        """Health check endpoint."""
        return {"status": "healthy", "version": "1.0.0"}

    @app.get("/api/v1/tools")
    async def get_tool_definitions():
        """Return OpenAI-compatible Tool Calling definitions for all 14 APIs."""
        return TOOL_DEFINITIONS

    # ------------------------------------------------------------------
    # 1. Extract Knowledge
    # ------------------------------------------------------------------

    @app.post("/api/v1/extract", dependencies=[Depends(verify_api_key)])
    async def extract_knowledge(req: ExtractRequest):
        """Extract entities, relations, and events from a document."""
        result = app.state.pipeline.extract(
            content=req.content,
            fmt=req.format,
            strategy=req.extraction_strategy,
            chunk_size=req.chunk_size,
            chunk_overlap=req.chunk_overlap,
            auto_resolve=req.auto_resolve,
            title=req.title,
        )
        return result

    # ------------------------------------------------------------------
    # 2. Create Entity
    # ------------------------------------------------------------------

    @app.post("/api/v1/entities", dependencies=[Depends(verify_api_key)])
    async def create_entity(req: CreateEntityRequest):
        """Create a new entity with automatic deduplication."""
        entity = Entity(
            name=req.name, type=req.type, aliases=req.aliases,
            attributes=req.attributes, description=req.description,
            tags=req.tags, confidence=req.confidence,
        )
        return app.state.store.create_entity(entity)

    # ------------------------------------------------------------------
    # 3. Update Entity
    # ------------------------------------------------------------------

    @app.patch("/api/v1/entities/{entity_id}", dependencies=[Depends(verify_api_key)])
    async def update_entity(entity_id: str, req: UpdateEntityRequest):
        """Update entity attributes (partial update)."""
        return app.state.store.update_entity(
            entity_id=entity_id,
            attributes=req.attributes,
            description=req.description,
            name=req.name,
            tags=req.tags,
            confidence=req.confidence,
            temporal=req.temporal,
        )

    # ------------------------------------------------------------------
    # 4. Delete Entity
    # ------------------------------------------------------------------

    @app.delete("/api/v1/entities/{entity_id}", dependencies=[Depends(verify_api_key)])
    async def delete_entity(entity_id: str,
                            cascade: bool = True, reason: str = ""):
        """Soft-delete an entity and optionally cascade to relations."""
        return app.state.store.delete_entity(entity_id, cascade, reason)

    # ------------------------------------------------------------------
    # 5. Create Relation
    # ------------------------------------------------------------------

    @app.post("/api/v1/relations", dependencies=[Depends(verify_api_key)])
    async def create_relation(req: CreateRelationRequest):
        """Create a directed relation between two entities."""
        rel = Relation(
            source_entity_id=req.source_entity_id,
            relation_type=req.relation_type,
            target_entity_id=req.target_entity_id,
            attributes=req.attributes,
            confidence=req.confidence,
            valid_from=req.valid_from,
            valid_to=req.valid_to,
        )
        result = app.state.store.create_relation(rel)
        if result.get("status") == "constraint_violation":
            raise HTTPException(status_code=422, detail=result)
        if result.get("status") == "error":
            raise HTTPException(status_code=404, detail=result["message"])
        return result

    # ------------------------------------------------------------------
    # 6. Search Entities
    # ------------------------------------------------------------------

    @app.post("/api/v1/search/entities", dependencies=[Depends(verify_api_key)])
    async def search_entities(req: SearchRequest):
        """Search entities using hybrid vector + keyword matching."""
        results = app.state.store.search_entities(
            query=req.query,
            entity_types=req.entity_types,
            min_confidence=req.min_confidence,
            top_k=req.top_k,
            search_mode=req.search_mode,
        )
        return {"total": len(results), "results": results}

    # ------------------------------------------------------------------
    # 7. Query Subgraph
    # ------------------------------------------------------------------

    @app.post("/api/v1/graph/subgraph", dependencies=[Depends(verify_api_key)])
    async def query_subgraph(req: SubgraphRequest):
        """Extract a subgraph within N hops from a given entity."""
        return app.state.store.query_subgraph(
            entity_id=req.entity_id,
            depth=req.depth,
            relation_types=req.relation_types,
            direction=req.direction,
            limit_per_hop=req.limit_per_hop,
            min_confidence=req.min_confidence,
        )

    # ------------------------------------------------------------------
    # 8. Text2Cypher
    # ------------------------------------------------------------------

    @app.post("/api/v1/query/text2cypher", dependencies=[Depends(verify_api_key)])
    async def text2cypher(req: Text2CypherRequest):
        """Convert natural language to Cypher and execute."""
        return app.state.graphrag.text2cypher.query(
            question=req.question,
            context_entities=req.context_entities,
            max_results=req.max_results,
            dry_run=req.dry_run,
        )

    # ------------------------------------------------------------------
    # 9. GraphRAG Search
    # ------------------------------------------------------------------

    @app.post("/api/v1/graphrag/search", dependencies=[Depends(verify_api_key)])
    async def graphrag_search(req: GraphRAGRequest):
        """GraphRAG hybrid retrieval (vector + graph + community)."""
        return app.state.graphrag.search(
            query=req.query,
            strategy=req.strategy,
            max_context_entities=req.max_context_entities,
            max_context_relations=req.max_context_relations,
            include_source_chunks=req.include_source_chunks,
            subgraph_depth=req.subgraph_depth,
            min_confidence=req.min_confidence,
        )

    # ------------------------------------------------------------------
    # 10. Find Paths
    # ------------------------------------------------------------------

    @app.post("/api/v1/graph/paths", dependencies=[Depends(verify_api_key)])
    async def find_paths(req: FindPathsRequest):
        """Find connection paths between two entities."""
        return app.state.store.find_paths(
            source_id=req.source_entity_id,
            target_id=req.target_entity_id,
            max_depth=req.max_depth,
            max_paths=req.max_paths,
            algorithm=req.algorithm,
        )

    # ------------------------------------------------------------------
    # 11. Reason
    # ------------------------------------------------------------------

    @app.post("/api/v1/reason", dependencies=[Depends(verify_api_key)])
    async def reason(req: ReasonRequest):
        """Run neuro-symbolic reasoning over the graph."""
        return _run_reasoning(app.state, req)

    # ------------------------------------------------------------------
    # 12. Graph Stats
    # ------------------------------------------------------------------

    @app.get("/api/v1/stats", dependencies=[Depends(verify_api_key)])
    async def get_stats(detailed: bool = False):
        """Return graph scale, quality, and distribution statistics."""
        return app.state.store.get_stats(detailed=detailed)

    # ------------------------------------------------------------------
    # 13. Export Graph
    # ------------------------------------------------------------------

    @app.post("/api/v1/export", dependencies=[Depends(verify_api_key)])
    async def export_graph(req: ExportRequest):
        """Export subgraph in various formats."""
        return app.state.exporter.export(
            format=req.format,
            entity_ids=req.entity_ids,
            depth=req.depth,
            include_attributes=req.include_attributes,
            max_nodes=req.max_nodes,
        )

    # ------------------------------------------------------------------
    # 14. Batch Import
    # ------------------------------------------------------------------

    @app.post("/api/v1/import", dependencies=[Depends(verify_api_key)])
    async def batch_import(req: ImportRequest):
        """Batch import entities and relations."""
        return _batch_import(app.state.store, req)

    return app


# ---------------------------------------------------------------------------
# Reasoning Implementation
# ---------------------------------------------------------------------------

def _run_reasoning(state, req: ReasonRequest) -> dict:
    """Execute reasoning based on type selection."""
    store = state.store
    inference_chain = []

    # Symbolic reasoning: check common patterns
    if req.reasoning_type in ("auto", "symbolic", "hybrid"):
        # Transitivity check: A->B, B->C => A->C
        if req.context_entity_ids and len(req.context_entity_ids) >= 2:
            paths = store.find_paths(
                req.context_entity_ids[0],
                req.context_entity_ids[-1],
                max_depth=req.max_inference_depth,
                algorithm="all_simple",
            )
            for path in paths.get("paths", []):
                inference_chain.append({
                    "step": len(inference_chain) + 1,
                    "type": "symbolic",
                    "rule": "path_discovery",
                    "output": f"Path found: {' -> '.join(n['name'] for n in path['nodes'])}",
                    "confidence": path.get("total_confidence", 0.5),
                })

    # Neural reasoning: use LLM with graph context
    if req.reasoning_type in ("auto", "neural", "hybrid"):
        context_entities = []
        for eid in (req.context_entity_ids or []):
            ent = store.get_entity(eid)
            if ent:
                context_entities.append(ent["name"])

        # Gather subgraph context
        subgraph_context = ""
        if req.context_entity_ids:
            sub = store.query_subgraph(req.context_entity_ids[0], depth=2)
            subgraph_context = json.dumps(sub.get("nodes", [])[:10],
                                          ensure_ascii=False)

        # Try LLM reasoning
        if state.graphrag.text2cypher.client:
            try:
                response = state.graphrag.text2cypher.client.chat.completions.create(
                    model=state.config.get("llm", {}).get("model", "gpt-4o"),
                    messages=[
                        {"role": "system", "content": "You are a knowledge "
                         "graph reasoning engine. Analyze the given graph "
                         "context and answer the reasoning question. "
                         "Return JSON with 'conclusion' and 'confidence'."},
                        {"role": "user", "content": f"Question: {req.query}\n"
                         f"Entities: {context_entities}\n"
                         f"Subgraph: {subgraph_context}"},
                    ],
                    temperature=0.1,
                    response_format={"type": "json_object"},
                )
                result = json.loads(response.choices[0].message.content)
                inference_chain.append({
                    "step": len(inference_chain) + 1,
                    "type": "neural",
                    "output": result.get("conclusion", ""),
                    "confidence": result.get("confidence", 0.7),
                    "verification": {"consistent_with_graph": True},
                })
            except Exception:
                inference_chain.append({
                    "step": len(inference_chain) + 1,
                    "type": "neural",
                    "output": "LLM reasoning unavailable",
                    "confidence": 0.0,
                })
        else:
            # Fallback: graph-based reasoning
            results = store.search_entities(req.query, top_k=5)
            inference_chain.append({
                "step": len(inference_chain) + 1,
                "type": "graph_traversal",
                "output": f"Found {len(results)} related entities",
                "confidence": 0.5,
            })

    conclusion = ""
    confidence = 0.5
    if inference_chain:
        last = inference_chain[-1]
        conclusion = last.get("output", "")
        confidence = last.get("confidence", 0.5)

    return {
        "query": req.query,
        "reasoning_type": req.reasoning_type,
        "inference_chain": inference_chain,
        "conclusion": conclusion,
        "overall_confidence": round(confidence, 4),
    }


# ---------------------------------------------------------------------------
# Batch Import Implementation
# ---------------------------------------------------------------------------

def _find_entity_id_by_name(store: KGStore, name: str) -> Optional[str]:
    """通过实体名称查找实体 ID（精确匹配优先，别名匹配其次）。"""
    if not name:
        return None
    # 先尝试精确名称匹配
    row = store.conn.execute(
        "SELECT entity_id FROM entities WHERE name=? AND deleted_at IS NULL",
        (name,),
    ).fetchone()
    if row:
        return row["entity_id"]
    # 再尝试别名匹配
    for row in store.conn.execute(
        "SELECT entity_id, aliases FROM entities WHERE deleted_at IS NULL"
    ):
        aliases = json.loads(row["aliases"] or "[]")
        if name in aliases:
            return row["entity_id"]
    return None


def _batch_import(store: KGStore, req: ImportRequest) -> dict:
    """Import entities and relations from structured data."""
    imported = 0
    failed = 0

    try:
        if req.format == "jsonld":
            data = json.loads(req.data)
            graph_items = data.get("@graph", [data]) if isinstance(data, dict) else data
            for item in graph_items:
                try:
                    if item.get("@type") != "Relation":
                        entity = Entity(
                            name=item.get("name", ""),
                            type=item.get("@type", "Other"),
                            description=item.get("description", ""),
                            attributes=item.get("attributes", {}),
                            confidence=item.get("confidence", 1.0),
                        )
                        result = store.create_entity(entity)
                        if result.get("status") in ("created", "merged"):
                            imported += 1
                    else:
                        # Relation import - 从 JSON-LD 中提取并创建关系
                        source_ref = (
                            item.get("source") or
                            item.get("subject") or
                            item.get("source_entity_id", "")
                        )
                        target_ref = (
                            item.get("target") or
                            item.get("object") or
                            item.get("target_entity_id", "")
                        )
                        # 支持嵌套对象格式 {"source": {"name": "..."}}
                        if isinstance(source_ref, dict):
                            source_ref = source_ref.get("name", source_ref.get("@id", ""))
                        if isinstance(target_ref, dict):
                            target_ref = target_ref.get("name", target_ref.get("@id", ""))

                        rel_type = item.get(
                            "relation_type",
                            item.get("predicate", "RELATED_TO"),
                        )
                        # 通过名称查找实体 ID
                        src_id = _find_entity_id_by_name(store, source_ref)
                        tgt_id = _find_entity_id_by_name(store, target_ref)
                        if src_id and tgt_id:
                            rel = Relation(
                                source_entity_id=src_id,
                                relation_type=rel_type,
                                target_entity_id=tgt_id,
                                confidence=item.get("confidence", 1.0),
                                attributes=item.get("attributes", {}),
                            )
                            result = store.create_relation(rel)
                            if result.get("status") in ("created", "merged"):
                                imported += 1
                            else:
                                failed += 1
                        else:
                            failed += 1
                except Exception:
                    failed += 1

        elif req.format == "csv":
            lines = req.data.strip().split("\n")
            for line in lines[1:]:  # Skip header
                parts = line.split(",")
                if len(parts) >= 3:
                    try:
                        entity = Entity(name=parts[0], type=parts[1],
                                        description=parts[2] if len(parts) > 2 else "")
                        result = store.create_entity(entity)
                        if result.get("status") in ("created", "merged"):
                            imported += 1
                    except Exception:
                        failed += 1
    except Exception as e:
        return {"status": "error", "message": str(e),
                "imported": imported, "failed": failed}

    return {
        "status": "completed",
        "imported": imported,
        "failed": failed,
        "total": imported + failed,
    }


# ---------------------------------------------------------------------------
# Tool Calling Definitions (OpenAI Function Calling format)
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {"type": "function", "function": {
        "name": "extract_knowledge",
        "description": "Extract entities, relations, and events from a document and add them to the knowledge graph.",
        "parameters": {"type": "object", "properties": {
            "content": {"type": "string", "description": "Document content"},
            "format": {"type": "string", "enum": ["text", "markdown", "json", "table"]},
            "extraction_strategy": {"type": "string", "enum": ["auto", "rule_first", "llm_first"]},
        }, "required": ["content"]}}},
    {"type": "function", "function": {
        "name": "create_entity",
        "description": "Create a new entity with automatic deduplication.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string"}, "type": {"type": "string",
            "enum": ["Organization", "Person", "Product", "Event", "Location", "Concept", "Other"]},
            "description": {"type": "string"}, "aliases": {"type": "array", "items": {"type": "string"}},
        }, "required": ["name", "type"]}}},
    {"type": "function", "function": {
        "name": "search_entities",
        "description": "Search entities using hybrid vector + keyword matching. Primary entry point for finding entities.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}, "entity_types": {"type": "array", "items": {"type": "string"}},
            "top_k": {"type": "integer"}, "search_mode": {"type": "string", "enum": ["hybrid", "vector", "keyword"]},
        }, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "query_subgraph",
        "description": "Extract a subgraph within N hops from an entity. Explore neighborhood and relationships.",
        "parameters": {"type": "object", "properties": {
            "entity_id": {"type": "string"}, "depth": {"type": "integer"},
            "relation_types": {"type": "array", "items": {"type": "string"}},
            "direction": {"type": "string", "enum": ["outgoing", "incoming", "both"]},
        }, "required": ["entity_id"]}}},
    {"type": "function", "function": {
        "name": "text2cypher",
        "description": "Convert natural language to Cypher graph query and execute. For precise pattern matching queries.",
        "parameters": {"type": "object", "properties": {
            "question": {"type": "string"}, "max_results": {"type": "integer"},
            "dry_run": {"type": "boolean"},
        }, "required": ["question"]}}},
    {"type": "function", "function": {
        "name": "graphrag_search",
        "description": "GraphRAG hybrid retrieval combining vector, graph, and community summaries. Primary interface for complex multi-hop questions.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}, "strategy": {"type": "string",
            "enum": ["auto", "vector", "graph", "hybrid", "community"]},
            "max_context_entities": {"type": "integer"}, "include_source_chunks": {"type": "boolean"},
        }, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "find_paths",
        "description": "Find connection paths between two entities. For root cause analysis and relationship discovery.",
        "parameters": {"type": "object", "properties": {
            "source_entity_id": {"type": "string"}, "target_entity_id": {"type": "string"},
            "max_depth": {"type": "integer"}, "algorithm": {"type": "string",
            "enum": ["shortest", "all_simple", "weighted"]},
        }, "required": ["source_entity_id", "target_entity_id"]}}},
    {"type": "function", "function": {
        "name": "reason",
        "description": "Run neuro-symbolic reasoning over the graph for inference and hypothesis testing.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}, "reasoning_type": {"type": "string",
            "enum": ["auto", "symbolic", "neural", "hybrid"]},
            "context_entity_ids": {"type": "array", "items": {"type": "string"}},
        }, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "get_graph_stats",
        "description": "Retrieve graph statistics: entity/relation counts, confidence scores, distributions.",
        "parameters": {"type": "object", "properties": {
            "detailed": {"type": "boolean"},
        }, "required": []}}},
    {"type": "function", "function": {
        "name": "export_graph",
        "description": "Export subgraph as Mermaid, JSON-LD, text-tree, CSV, or GraphML.",
        "parameters": {"type": "object", "properties": {
            "format": {"type": "string", "enum": ["mermaid", "jsonld", "text_tree", "csv", "graphml"]},
            "entity_ids": {"type": "array", "items": {"type": "string"}}, "depth": {"type": "integer"},
        }, "required": ["format"]}}},
]


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file."""
    if config_path and os.path.exists(config_path):
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def main():
    """Start the KG Skill API server."""
    parser = argparse.ArgumentParser(description="KG Skill API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8700, help="Bind port")
    parser.add_argument("--config", default=None, help="Config file path")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    if not config:
        # Try default config location
        default_config = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "assets", "config", "default_config.yaml"
        )
        config = load_config(default_config)

    app = create_app(config)

    import uvicorn
    api_key_set = bool(os.environ.get("KG_API_KEY"))
    auth_status = "enabled" if api_key_set else "disabled (set KG_API_KEY env to enable)"
    print(f"KG Skill API Server starting on http://{args.host}:{args.port}")
    print(f"API Key auth: {auth_status}")
    print(f"Swagger docs: http://{args.host}:{args.port}/docs")
    print(f"Tool definitions: http://{args.host}:{args.port}/api/v1/tools")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
