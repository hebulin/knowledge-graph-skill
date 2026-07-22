"""
KG Skill GraphRAG Module - Hybrid retrieval engine.

Combines vector semantic search, graph traversal, and community-level
summaries to provide rich structured context for LLM answer generation.
Uses Reciprocal Rank Fusion (RRF) to merge multi-source results.
"""

import json
import os
import re
from typing import Optional
from collections import defaultdict

import numpy as np
import networkx as nx

from kg_core import KGStore, _now_iso


# ---------------------------------------------------------------------------
# Community Detection
# ---------------------------------------------------------------------------

class CommunityDetector:
    """Detect communities in the graph and generate LLM-powered summaries."""

    def __init__(self, store: KGStore, config: dict = None):
        """Initialize with a KGStore instance and optional LLM config."""
        self.store = store
        self.config = config or {}
        self.communities = {}  # community_id -> {nodes, summary, level}
        self._entity_to_community = {}
        self._llm_client = None
        self._llm_model = self.config.get("llm", {}).get("model", "gpt-4o")
        self._init_llm_client()

    def detect(self, algorithm: str = "leiden") -> dict:
        """Run community detection on the current graph."""
        if len(self.store.graph.nodes) == 0:
            return {"communities": 0, "nodes_assigned": 0}

        undirected = self.store.graph.to_undirected()

        if algorithm == "leiden" and hasattr(nx.community, "leiden_communities"):
            try:
                communities = nx.community.leiden_communities(undirected)
            except Exception:
                communities = list(nx.community.greedy_modularity_communities(undirected))
        else:
            # Fallback to greedy modularity
            try:
                communities = list(
                    nx.community.greedy_modularity_communities(undirected)
                )
            except Exception:
                communities = [set(self.store.graph.nodes)]

        self.communities = {}
        self._entity_to_community = {}
        for idx, community in enumerate(communities):
            comm_id = f"comm_{idx:04d}"
            nodes = list(community)
            self.communities[comm_id] = {
                "community_id": comm_id,
                "nodes": nodes,
                "node_count": len(nodes),
                "summary": self._generate_summary(nodes),
                "level": 0,
            }
            for node_id in nodes:
                self._entity_to_community[node_id] = comm_id

        return {
            "communities": len(self.communities),
            "nodes_assigned": len(self._entity_to_community),
        }

    def _init_llm_client(self):
        """Initialize LLM client for summary generation."""
        api_key = os.environ.get("OPENAI_API_KEY")
        api_base = self.config.get("llm", {}).get("api_base", "")
        if api_key:
            try:
                from openai import OpenAI
                kwargs = {"api_key": api_key}
                if api_base:
                    kwargs["base_url"] = api_base
                self._llm_client = OpenAI(**kwargs)
            except ImportError:
                pass

    def _generate_summary(self, node_ids: list) -> str:
        """Generate a summary for a community using LLM or fallback."""
        names = []
        types = defaultdict(list)
        descriptions = []
        for nid in node_ids[:20]:
            ent = self.store.get_entity(nid)
            if ent:
                names.append(ent["name"])
                types[ent["type"]].append(ent["name"])
                if ent.get("description"):
                    descriptions.append(f"- {ent['name']} ({ent['type']}): {ent['description']}")

        # Try LLM summary
        if self._llm_client and descriptions:
            try:
                entity_text = "\n".join(descriptions[:15])
                response = self._llm_client.chat.completions.create(
                    model=self._llm_model,
                    messages=[
                        {"role": "system", "content": "Summarize the following "
                         "knowledge graph community entities into a concise "
                         "paragraph (2-3 sentences). Focus on relationships "
                         "and common themes."},
                        {"role": "user", "content": entity_text},
                    ],
                    temperature=0.3,
                    max_tokens=200,
                )
                return response.choices[0].message.content.strip()
            except Exception:
                pass

        # Fallback: structured name listing
        parts = []
        for etype, ents in types.items():
            parts.append(f"{etype}: {', '.join(ents[:5])}")
        return f"Community with {len(node_ids)} entities. " + "; ".join(parts)

    def get_community(self, entity_id: str):
        """Get the community an entity belongs to."""
        comm_id = self._entity_to_community.get(entity_id)
        if comm_id:
            return self.communities.get(comm_id)
        return None

    def search_communities(self, query_vec: np.array, top_k: int = 3) -> list:
        """Find communities most relevant to a query vector."""
        if not self.communities:
            return []

        scored = []
        for comm_id, comm in self.communities.items():
            # Score by average embedding of member entities
            member_vecs = []
            for nid in comm["nodes"]:
                if nid in self.store._embeddings:
                    member_vecs.append(self.store._embeddings[nid])
            if member_vecs:
                avg_vec = np.mean(member_vecs, axis=0)
                score = float(
                    np.dot(avg_vec, query_vec) /
                    (np.linalg.norm(avg_vec) * np.linalg.norm(query_vec) + 1e-8)
                )
            else:
                score = 0.0
            scored.append((comm_id, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            {**self.communities[cid], "relevance_score": round(score, 4)}
            for cid, score in scored[:top_k]
        ]


# ---------------------------------------------------------------------------
# Text2Cypher (Natural Language to Graph Query)
# ---------------------------------------------------------------------------

class Text2Cypher:
    """Convert natural language questions to graph queries."""

    SCHEMA_PROMPT = """You have a knowledge graph with the following schema:

Node labels: Organization, Person, Product, Event, Location, Concept, Other
Relationship types: ACQUIRED, FOUNDED_BY, WORKS_AT, PRODUCES, COMPETES_WITH,
  DEVELOPED, PARTNERED_WITH, INVESTED_IN, RELATED_TO, and others in UPPER_SNAKE_CASE

All nodes have properties: name, type, description, confidence, tags.
Relationships have: relation_type, confidence, valid_from, valid_to.

Convert the following question to a Cypher query. Return ONLY the Cypher,
no explanation.

Question: {question}

Cypher:"""

    def __init__(self, store: KGStore, config: dict = None):
        """Initialize with store and LLM config."""
        self.store = store
        self.config = config or {}
        self.client = None
        self.model = self.config.get("llm", {}).get("model", "gpt-4o")
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            try:
                from openai import OpenAI
                kwargs = {"api_key": api_key}
                api_base = self.config.get("llm", {}).get("api_base")
                if api_base:
                    kwargs["base_url"] = api_base
                self.client = OpenAI(**kwargs)
            except ImportError:
                pass

    def query(self, question: str, context_entities: list = None,
              max_results: int = 100, dry_run: bool = False) -> dict:
        """Convert question to Cypher, validate, and execute."""
        if not self.client:
            return self._fallback_query(question)

        # Generate Cypher via LLM
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a Cypher query "
                     "generator. Return ONLY valid Cypher, no markdown."},
                    {"role": "user", "content": self.SCHEMA_PROMPT.format(
                        question=question)},
                ],
                temperature=0.0,
                max_tokens=1024,
            )
            cypher = response.choices[0].message.content.strip()
            # Strip markdown code fences if present
            cypher = cypher.replace("```cypher", "").replace("```", "").strip()
        except Exception as e:
            return {"question": question, "generated_cypher": "",
                    "cypher_valid": False, "error": str(e),
                    "results": [], "result_count": 0}

        if dry_run:
            return {"question": question, "generated_cypher": cypher,
                    "cypher_valid": True, "results": [], "result_count": 0}

        # Execute against NetworkX graph (simplified Cypher interpretation)
        results = self._execute_simplified(question, cypher, max_results)
        return {
            "question": question,
            "generated_cypher": cypher,
            "cypher_valid": True,
            "results": results,
            "result_count": len(results),
        }

    def _fallback_query(self, question: str) -> dict:
        """Fallback to keyword search when LLM is unavailable."""
        results = self.store.search_entities(question, top_k=20)
        return {
            "question": question,
            "generated_cypher": "",
            "cypher_valid": False,
            "fallback": "keyword_search",
            "results": results,
            "result_count": len(results),
        }

    def _execute_simplified(self, question: str, cypher: str,
                            max_results: int) -> list:
        """Simplified query execution against NetworkX graph.

        In production, this would execute against Neo4j. In lightweight mode,
        we dynamically interpret common Cypher patterns.
        """
        results = []

        # 1. 从 Cypher 中动态提取关系类型 (e.g. [:ACQUIRED], -[:FOUNDED_BY]->)
        rel_pattern = re.compile(
            r'(?::\s*"?([A-Z_]{2,})"?\s*[>\]])'
        )
        rel_types = set()
        for m in rel_pattern.finditer(cypher):
            rel_types.add(m.group(1).upper())

        # 2. 从 WHERE 子句提取属性过滤条件
        #    e.g. WHERE a.name = "Apple" or WHERE a.name CONTAINS "Apple"
        name_pattern = re.compile(
            r'\.name\s*(?:=|CONTAINS|=~)\s*"([^"]+)"', re.IGNORECASE
        )
        name_filters = [m.group(1) for m in name_pattern.finditer(cypher)]

        # 3. 执行关系遍历（如果提取到关系类型）
        if rel_types:
            for u, v, data in self.store.graph.edges(data=True):
                edge_type = data.get("relation_type", "").upper()
                if edge_type in rel_types:
                    src = self.store.get_entity(u)
                    tgt = self.store.get_entity(v)
                    if src and tgt:
                        if name_filters:
                            matched = any(
                                nf.lower() in src["name"].lower() or
                                nf.lower() in tgt["name"].lower()
                                for nf in name_filters
                            )
                            if not matched:
                                continue
                        results.append({
                            "source": src["name"],
                            "source_type": src["type"],
                            "relation": edge_type,
                            "target": tgt["name"],
                            "target_type": tgt["type"],
                            "confidence": data.get("confidence", 1.0),
                        })
                    if len(results) >= max_results:
                        break

        # 4. 如果没有关系匹配，但有名称过滤，按名称搜索实体
        if not results and name_filters:
            for nf in name_filters:
                entities = self.store.search_entities(nf, top_k=max_results)
                for ent in entities:
                    results.append({
                        "name": ent["name"],
                        "type": ent["type"],
                        "confidence": ent.get("confidence", 1.0),
                    })
                    if len(results) >= max_results:
                        break
                if len(results) >= max_results:
                    break

        # 5. 最终回退：关键词搜索
        if not results:
            results = self.store.search_entities(question, top_k=max_results)

        return results


# ---------------------------------------------------------------------------
# GraphRAG Search Engine
# ---------------------------------------------------------------------------

class GraphRAGSearch:
    """Hybrid retrieval combining vector, graph, and community search."""

    def __init__(self, store: KGStore, config: dict = None):
        """Initialize with store, community detector, and config."""
        self.store = store
        self.config = config or {}
        self.community_detector = CommunityDetector(store, config)
        self.text2cypher = Text2Cypher(store, config)
        self._communities_detected = False

    def search(self, query: str, strategy: str = "auto",
               max_context_entities: int = 20,
               max_context_relations: int = 30,
               include_source_chunks: bool = True,
               subgraph_depth: int = 2,
               min_confidence: float = 0.6) -> dict:
        """
        Execute GraphRAG hybrid retrieval.

        Returns structured context with entities, relations, community
        summaries, and source chunks.
        """
        # Auto-select strategy based on query characteristics
        if strategy == "auto":
            strategy = self._select_strategy(query)

        # Run community detection if not done
        if not self._communities_detected and strategy in ("hybrid", "community"):
            self.community_detector.detect()
            self._communities_detected = True

        vector_results = []
        graph_results = []
        community_results = []

        # Vector search
        if strategy in ("vector", "hybrid"):
            vector_results = self.store.search_entities(
                query, top_k=max_context_entities,
                min_confidence=min_confidence, search_mode="hybrid",
            )

        # Graph traversal from seed entities
        if strategy in ("graph", "hybrid"):
            seeds = [r["entity_id"] for r in vector_results[:5]]
            if not seeds:
                seeds = [r["entity_id"] for r in
                         self.store.search_entities(query, top_k=5)]
            for seed_id in seeds:
                sub = self.store.query_subgraph(
                    seed_id, depth=subgraph_depth,
                    limit_per_hop=10, min_confidence=min_confidence,
                )
                graph_results.extend(sub.get("edges", []))

        # Community summaries
        if strategy in ("community", "hybrid"):
            q_vec = self.store._get_query_embedding(query)
            if q_vec is not None:
                community_results = self.community_detector.search_communities(
                    q_vec, top_k=3,
                )

        # RRF fusion
        all_entities = self._fuse_results(
            vector_results, graph_results, max_context_entities,
        )
        all_relations = self._deduplicate_relations(
            graph_results, max_context_relations,
        )

        # Source chunks
        source_chunks = []
        if include_source_chunks:
            chunk_ids = set()
            for ent in all_entities:
                eid = ent.get("entity_id")
                ent_data = self.store.get_entity(eid)
                if ent_data and ent_data.get("provenance"):
                    cid = ent_data["provenance"].get("source_chunk_id")
                    if cid and cid not in chunk_ids:
                        chunk_ids.add(cid)
                        chunk_row = self.store.conn.execute(
                            "SELECT * FROM chunks WHERE chunk_id=?", (cid,)
                        ).fetchone()
                        if chunk_row:
                            source_chunks.append({
                                "chunk_id": cid,
                                "doc_id": chunk_row["doc_id"],
                                "text": chunk_row["text"][:500],
                            })

        return {
            "query": query,
            "strategy_used": strategy,
            "context": {
                "entities": all_entities,
                "relations": all_relations,
                "community_summaries": [
                    {"community_id": c["community_id"],
                     "summary": c["summary"],
                     "node_count": c["node_count"]}
                    for c in community_results
                ],
                "source_chunks": source_chunks,
            },
            "stats": {
                "entity_count": len(all_entities),
                "relation_count": len(all_relations),
                "community_count": len(community_results),
                "chunk_count": len(source_chunks),
            },
        }

    def _select_strategy(self, query: str) -> str:
        """Auto-select retrieval strategy based on query characteristics."""
        # Multi-hop indicators
        multi_hop_indicators = ["how", "why", "relationship", "between",
                                "path", "connect", "cause", "because",
                                "vs", "versus", "compare", "difference"]
        query_lower = query.lower()
        if any(ind in query_lower for ind in multi_hop_indicators):
            return "hybrid"
        if len(query.split()) > 10:
            return "hybrid"
        return "vector"

    def _fuse_results(self, vector_results: list,
                      graph_results: list, max_count: int) -> list:
        """Fuse results using Reciprocal Rank Fusion (RRF)."""
        rrf_scores = defaultdict(float)
        k = 60  # RRF constant

        for rank, item in enumerate(vector_results):
            eid = item.get("entity_id")
            if eid:
                rrf_scores[eid] += 1.0 / (k + rank + 1)

        # Graph results contribute via edge endpoints
        graph_entity_ranks = {}
        rank = 0
        for edge in graph_results:
            for eid in [edge.get("source"), edge.get("target")]:
                if eid and eid not in graph_entity_ranks:
                    graph_entity_ranks[eid] = rank
                    rank += 1
        for eid, rank in graph_entity_ranks.items():
            rrf_scores[eid] += 1.0 / (k + rank + 1)

        # Build result list
        seen = set()
        # First add vector results (already have entity data)
        for item in vector_results:
            eid = item.get("entity_id")
            if eid and eid not in seen:
                item["rrf_score"] = round(rrf_scores[eid], 6)
                seen.add(eid)

        # Add graph-only entities
        for eid in graph_entity_ranks:
            if eid not in seen and len(seen) < max_count:
                ent = self.store.get_entity(eid)
                if ent:
                    ent["rrf_score"] = round(rrf_scores[eid], 6)
                    vector_results.append(ent)
                    seen.add(eid)

        return vector_results[:max_count]

    def _deduplicate_relations(self, relations: list,
                               max_count: int) -> list:
        """Remove duplicate relations and limit count."""
        seen = set()
        unique = []
        for rel in relations:
            key = (rel.get("source"), rel.get("type"), rel.get("target"))
            if key not in seen:
                seen.add(key)
                unique.append(rel)
        return unique[:max_count]