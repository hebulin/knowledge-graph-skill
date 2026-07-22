"""
KG Skill Core Module - Data model, storage, and CRUD operations.

Provides the foundational layer for the knowledge graph skill:
- Data classes for Entity, Relation, Event, Document, Chunk
- SQLite + NetworkX hybrid storage (lightweight mode, zero external DB)
- CRUD operations with soft-delete and temporal versioning
- 4-stage entity resolution (exact -> alias -> semantic -> LLM)
- In-memory vector index using NumPy cosine similarity
"""

import json
import sqlite3
import hashlib
import os
import time
import math
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import networkx as nx


# ---------------------------------------------------------------------------
# ULID generator (simplified - sortable unique ID without external dependency)
# ---------------------------------------------------------------------------

def _generate_id(prefix: str) -> str:
    """Generate a sortable unique ID with the given prefix."""
    ts = int(time.time() * 1000)
    rand = os.urandom(10).hex()
    return f"{prefix}_{ts:x}{rand}"


def _now_iso() -> str:
    """Return current UTC time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class Entity:
    """Knowledge graph entity node."""
    entity_id: str = ""
    name: str = ""
    type: str = "Other"
    aliases: list = field(default_factory=list)
    attributes: dict = field(default_factory=dict)
    description: str = ""
    confidence: float = 1.0
    tags: list = field(default_factory=list)
    provenance: dict = field(default_factory=dict)
    lifecycle_state: str = "active"
    tenant_id: str = "default"
    created_at: str = ""
    updated_at: str = ""
    deleted_at: Optional[str] = None
    _embedding: Optional[list] = None  # not persisted in SQLite, held in memory

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("_embedding", None)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Entity":
        d = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**d)


@dataclass
class Relation:
    """Knowledge graph directed edge between two entities."""
    relation_id: str = ""
    source_entity_id: str = ""
    relation_type: str = ""
    target_entity_id: str = ""
    attributes: dict = field(default_factory=dict)
    confidence: float = 1.0
    direction: str = "directed"
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None
    provenance: dict = field(default_factory=dict)
    tenant_id: str = "default"
    created_at: str = ""
    updated_at: str = ""
    deleted_at: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Relation":
        d = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**d)


@dataclass
class Event:
    """Event node for expressing N-ary relations (multi-entity events)."""
    event_id: str = ""
    event_type: str = "Other"
    participants: list = field(default_factory=list)
    attributes: dict = field(default_factory=dict)
    occurred_at: str = ""
    description: str = ""
    confidence: float = 1.0
    provenance: dict = field(default_factory=dict)
    tenant_id: str = "default"
    created_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Document:
    """Source document record for provenance tracking."""
    doc_id: str = ""
    title: str = ""
    format: str = "text"
    content: str = ""
    tenant_id: str = "default"
    created_at: str = ""


@dataclass
class Chunk:
    """Text chunk extracted from a document, bound to triples for grounding."""
    chunk_id: str = ""
    doc_id: str = ""
    text: str = ""
    chunk_index: int = 0
    entities_referenced: list = field(default_factory=list)
    created_at: str = ""


# ---------------------------------------------------------------------------
# KGStore - Main storage and operations class
# ---------------------------------------------------------------------------

class KGStore:
    """
    Knowledge graph store with SQLite persistence and NetworkX graph engine.

    Lightweight mode: SQLite (metadata) + NetworkX (graph topology) + NumPy
    (vector similarity). No external database required.
    """

    def __init__(self, db_path: str = None, config: dict = None):
        """Initialize the KG store with SQLite backend and in-memory graph."""
        if db_path is None:
            home = os.path.expanduser("~")
            db_dir = os.path.join(home, ".knowledge-graph-skill")
            os.makedirs(db_dir, exist_ok=True)
            db_path = os.path.join(db_dir, "kg.db")

        self.db_path = db_path
        self.config = config or {}
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.graph = nx.MultiDiGraph()
        self._embeddings = {}  # entity_id -> np.array
        self._init_db()
        self._load_graph()

    def _init_db(self):
        """Create SQLite tables if they do not exist."""
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS entities (
            entity_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            type TEXT DEFAULT 'Other',
            aliases TEXT DEFAULT '[]',
            attributes TEXT DEFAULT '{}',
            description TEXT DEFAULT '',
            confidence REAL DEFAULT 1.0,
            tags TEXT DEFAULT '[]',
            provenance TEXT DEFAULT '{}',
            lifecycle_state TEXT DEFAULT 'active',
            tenant_id TEXT DEFAULT 'default',
            created_at TEXT,
            updated_at TEXT,
            deleted_at TEXT
        );
        CREATE TABLE IF NOT EXISTS relations (
            relation_id TEXT PRIMARY KEY,
            source_entity_id TEXT NOT NULL,
            relation_type TEXT NOT NULL,
            target_entity_id TEXT NOT NULL,
            attributes TEXT DEFAULT '{}',
            confidence REAL DEFAULT 1.0,
            direction TEXT DEFAULT 'directed',
            valid_from TEXT,
            valid_to TEXT,
            provenance TEXT DEFAULT '{}',
            tenant_id TEXT DEFAULT 'default',
            created_at TEXT,
            updated_at TEXT,
            deleted_at TEXT
        );
        CREATE TABLE IF NOT EXISTS events (
            event_id TEXT PRIMARY KEY,
            event_type TEXT,
            participants TEXT DEFAULT '[]',
            attributes TEXT DEFAULT '{}',
            occurred_at TEXT,
            description TEXT,
            confidence REAL,
            provenance TEXT DEFAULT '{}',
            tenant_id TEXT,
            created_at TEXT
        );
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            title TEXT,
            format TEXT,
            content TEXT,
            tenant_id TEXT,
            created_at TEXT
        );
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            doc_id TEXT,
            text TEXT,
            chunk_index INTEGER,
            entities_referenced TEXT DEFAULT '[]',
            created_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_entity_name ON entities(name);
        CREATE INDEX IF NOT EXISTS idx_entity_type ON entities(type);
        CREATE INDEX IF NOT EXISTS idx_relation_type ON relations(relation_type);
        """)
        self.conn.commit()

    def _load_graph(self):
        """Load all non-deleted entities and relations into NetworkX graph."""
        for row in self.conn.execute(
            "SELECT * FROM entities WHERE deleted_at IS NULL"
        ):
            self.graph.add_node(
                row["entity_id"],
                name=row["name"],
                type=row["type"],
                confidence=row["confidence"],
            )
        for row in self.conn.execute(
            "SELECT * FROM relations WHERE deleted_at IS NULL"
        ):
            self.graph.add_edge(
                row["source_entity_id"],
                row["target_entity_id"],
                relation_id=row["relation_id"],
                relation_type=row["relation_type"],
                confidence=row["confidence"],
            )

    # ------------------------------------------------------------------
    # Entity CRUD
    # ------------------------------------------------------------------

    def create_entity(self, entity: Entity) -> dict:
        """Create an entity, auto-triggering deduplication check."""
        # 4-stage entity resolution
        match = self._resolve_entity(entity)
        if match:
            return {
                "entity_id": match["entity_id"],
                "status": "merged",
                "deduplication": match,
            }

        if not entity.entity_id:
            entity.entity_id = _generate_id("ent")
        now = _now_iso()
        entity.created_at = now
        entity.updated_at = now

        self.conn.execute(
            """INSERT INTO entities VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                entity.entity_id, entity.name, entity.type,
                json.dumps(entity.aliases, ensure_ascii=False),
                json.dumps(entity.attributes, ensure_ascii=False),
                entity.description, entity.confidence,
                json.dumps(entity.tags, ensure_ascii=False),
                json.dumps(entity.provenance, ensure_ascii=False),
                entity.lifecycle_state, entity.tenant_id,
                entity.created_at, entity.updated_at, entity.deleted_at,
            ),
        )
        self.conn.commit()
        self.graph.add_node(
            entity.entity_id, name=entity.name,
            type=entity.type, confidence=entity.confidence,
        )
        return {"entity_id": entity.entity_id, "status": "created",
                "deduplication": {"matched": False}}

    def get_entity(self, entity_id: str) -> Optional[dict]:
        """Retrieve a single entity by ID."""
        row = self.conn.execute(
            "SELECT * FROM entities WHERE entity_id=? AND deleted_at IS NULL",
            (entity_id,),
        ).fetchone()
        if not row:
            return None
        return self._row_to_entity_dict(row)

    def search_entities(
        self, query: str, entity_types: list = None,
        min_confidence: float = 0.0, top_k: int = 10,
        search_mode: str = "hybrid",
    ) -> list:
        """Search entities by keyword and/or semantic similarity."""
        results = []
        query_lower = query.lower()

        for row in self.conn.execute(
            "SELECT * FROM entities WHERE deleted_at IS NULL"
        ):
            if row["confidence"] < min_confidence:
                continue
            if entity_types and row["type"] not in entity_types:
                continue

            name = row["name"] or ""
            desc = row["description"] or ""
            aliases = json.loads(row["aliases"] or "[]")

            # Keyword score
            kw_score = 0.0
            if query_lower in name.lower():
                kw_score = max(kw_score, 1.0)
            if query_lower in desc.lower():
                kw_score = max(kw_score, 0.6)
            for alias in aliases:
                if query_lower in alias.lower():
                    kw_score = max(kw_score, 0.9)

            # Vector score (if embeddings available)
            vec_score = 0.0
            if search_mode in ("hybrid", "vector") and self._embeddings:
                ent_id = row["entity_id"]
                if ent_id in self._embeddings:
                    q_vec = self._get_query_embedding(query)
                    if q_vec is not None:
                        vec_score = float(
                            np.dot(self._embeddings[ent_id], q_vec) /
                            (np.linalg.norm(self._embeddings[ent_id]) *
                             np.linalg.norm(q_vec) + 1e-8)
                        )

            if search_mode == "vector":
                score = vec_score
            elif search_mode == "keyword":
                score = kw_score
            else:  # hybrid
                score = 0.5 * kw_score + 0.5 * vec_score

            if score > 0:
                results.append({
                    "entity_id": row["entity_id"],
                    "name": name,
                    "type": row["type"],
                    "description": desc,
                    "confidence": row["confidence"],
                    "score": round(score, 4),
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def update_entity(
        self, entity_id: str, attributes: dict = None,
        description: str = None, name: str = None,
        tags: list = None, confidence: float = None,
        temporal: bool = False,
    ) -> dict:
        """Update entity fields (partial update, records audit trail)."""
        row = self.conn.execute(
            "SELECT * FROM entities WHERE entity_id=? AND deleted_at IS NULL",
            (entity_id,),
        ).fetchone()
        if not row:
            return {"status": "not_found"}

        prev_values = {}
        updates = []
        params = []

        if name is not None:
            updates.append("name=?")
            params.append(name)
            prev_values["name"] = row["name"]
        if description is not None:
            updates.append("description=?")
            params.append(description)
            prev_values["description"] = row["description"]
        if attributes is not None:
            existing = json.loads(row["attributes"] or "{}")
            if temporal:
                for k, v in attributes.items():
                    old_val = existing.get(k)
                    if old_val is not None:
                        hist_key = f"__history__{k}"
                        existing.setdefault(hist_key, []).append({
                            "value": old_val,
                            "valid_to": _now_iso(),
                        })
                    existing[k] = {"value": v, "valid_from": _now_iso()}
            else:
                existing.update(attributes)
            updates.append("attributes=?")
            params.append(json.dumps(existing, ensure_ascii=False))
            prev_values["attributes"] = row["attributes"]
        if tags is not None:
            updates.append("tags=?")
            params.append(json.dumps(tags, ensure_ascii=False))
            prev_values["tags"] = row["tags"]
        if confidence is not None:
            updates.append("confidence=?")
            params.append(confidence)
            prev_values["confidence"] = row["confidence"]

        if not updates:
            return {"status": "no_changes"}

        updates.append("updated_at=?")
        params.append(_now_iso())
        params.append(entity_id)

        self.conn.execute(
            f"UPDATE entities SET {', '.join(updates)} WHERE entity_id=?",
            params,
        )
        self.conn.commit()

        # Update graph node
        if name is not None and entity_id in self.graph:
            self.graph.nodes[entity_id]["name"] = name

        return {
            "entity_id": entity_id,
            "status": "updated",
            "updated_fields": list(prev_values.keys()),
            "previous_values": prev_values,
        }

    def delete_entity(
        self, entity_id: str, cascade: bool = True, reason: str = "",
    ) -> dict:
        """Soft-delete an entity and optionally cascade to relations."""
        now = _now_iso()
        cascade_count = 0

        self.conn.execute(
            "UPDATE entities SET deleted_at=?, lifecycle_state='deprecated' "
            "WHERE entity_id=?",
            (now, entity_id),
        )

        if cascade:
            for row in self.conn.execute(
                "SELECT relation_id FROM relations WHERE "
                "(source_entity_id=? OR target_entity_id=?) AND deleted_at IS NULL",
                (entity_id, entity_id),
            ):
                self.conn.execute(
                    "UPDATE relations SET deleted_at=? WHERE relation_id=?",
                    (now, row["relation_id"]),
                )
                cascade_count += 1

        if entity_id in self.graph:
            self.graph.remove_node(entity_id)

        self.conn.commit()
        return {
            "entity_id": entity_id,
            "status": "deleted",
            "cascade_deleted_relations": cascade_count,
            "deleted_at": now,
        }

    # ------------------------------------------------------------------
    # Relation CRUD
    # ------------------------------------------------------------------

    def create_relation(self, rel: Relation) -> dict:
        """Create a relation with constraint validation."""
        # Validate entities exist
        src = self.get_entity(rel.source_entity_id)
        tgt = self.get_entity(rel.target_entity_id)
        if not src or not tgt:
            return {"status": "error",
                    "message": "Source or target entity not found"}

        # Self-loop check
        if rel.source_entity_id == rel.target_entity_id:
            return {"status": "constraint_violation",
                    "errors": [{"constraint": "self_loop",
                               "message": "Self-loop not allowed"}]}

        if not rel.relation_id:
            rel.relation_id = _generate_id("rel")
        now = _now_iso()
        rel.created_at = now
        rel.updated_at = now

        self.conn.execute(
            """INSERT INTO relations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                rel.relation_id, rel.source_entity_id, rel.relation_type,
                rel.target_entity_id,
                json.dumps(rel.attributes, ensure_ascii=False),
                rel.confidence, rel.direction, rel.valid_from, rel.valid_to,
                json.dumps(rel.provenance, ensure_ascii=False),
                rel.tenant_id, rel.created_at, rel.updated_at, rel.deleted_at,
            ),
        )
        self.conn.commit()
        self.graph.add_edge(
            rel.source_entity_id, rel.target_entity_id,
            relation_id=rel.relation_id,
            relation_type=rel.relation_type,
            confidence=rel.confidence,
        )
        return {"relation_id": rel.relation_id, "status": "created"}

    def delete_relation(self, relation_id: str) -> dict:
        """Soft-delete a relation."""
        now = _now_iso()
        self.conn.execute(
            "UPDATE relations SET deleted_at=? WHERE relation_id=?",
            (now, relation_id),
        )
        self.conn.commit()
        # Remove from graph
        edges_to_remove = [
            (u, v, k) for u, v, k, d in self.graph.edges(keys=True, data=True)
            if d.get("relation_id") == relation_id
        ]
        for u, v, k in edges_to_remove:
            self.graph.remove_edge(u, v, k)
        return {"relation_id": relation_id, "status": "deleted",
                "deleted_at": now}

    # ------------------------------------------------------------------
    # Graph Query Operations
    # ------------------------------------------------------------------

    def get_neighbors(
        self, entity_id: str, relation_types: list = None,
        direction: str = "both", limit: int = 50,
    ) -> list:
        """Get 1-hop neighbors of an entity."""
        if entity_id not in self.graph:
            return []

        neighbors = []
        if direction in ("outgoing", "both"):
            for _, tgt, data in self.graph.edges(entity_id, data=True):
                if relation_types and data.get("relation_type") not in relation_types:
                    continue
                ent = self.get_entity(tgt)
                if ent:
                    neighbors.append({
                        "entity": ent, "relation_type": data.get("relation_type"),
                        "direction": "outgoing",
                        "confidence": data.get("confidence", 1.0),
                    })
        if direction in ("incoming", "both"):
            for src, _, data in self.graph.in_edges(entity_id, data=True):
                if relation_types and data.get("relation_type") not in relation_types:
                    continue
                ent = self.get_entity(src)
                if ent:
                    neighbors.append({
                        "entity": ent, "relation_type": data.get("relation_type"),
                        "direction": "incoming",
                        "confidence": data.get("confidence", 1.0),
                    })
        return neighbors[:limit]

    def query_subgraph(
        self, entity_id: str, depth: int = 2,
        relation_types: list = None, direction: str = "both",
        limit_per_hop: int = 50, min_confidence: float = 0.0,
    ) -> dict:
        """Extract a subgraph within N hops from the given entity."""
        if entity_id not in self.graph:
            return {"nodes": [], "edges": [], "stats": {"node_count": 0,
                    "edge_count": 0, "max_depth_reached": 0}}

        # BFS traversal
        visited = {entity_id: 0}
        nodes = []
        edges = []

        ent = self.get_entity(entity_id)
        if ent:
            nodes.append({**ent, "depth": 0})

        frontier = [entity_id]
        for hop in range(1, depth + 1):
            next_frontier = []
            for node_id in frontier:
                edges_iter = []
                if direction in ("outgoing", "both"):
                    edges_iter.extend(
                        (n_id, tgt, d) for n_id, tgt, d
                        in self.graph.edges(node_id, data=True)
                    )
                if direction in ("incoming", "both"):
                    edges_iter.extend(
                        (src, n_id, d) for src, n_id, d
                        in self.graph.in_edges(node_id, data=True)
                    )

                count = 0
                for src, tgt, data in edges_iter:
                    if count >= limit_per_hop:
                        break
                    rt = data.get("relation_type")
                    if relation_types and rt not in relation_types:
                        continue
                    if data.get("confidence", 1.0) < min_confidence:
                        continue

                    edges.append({
                        "relation_id": data.get("relation_id"),
                        "source": src, "type": rt, "target": tgt,
                        "confidence": data.get("confidence", 1.0),
                    })

                    neighbor_id = tgt if src == node_id else src
                    if neighbor_id not in visited:
                        visited[neighbor_id] = hop
                        next_frontier.append(neighbor_id)
                        n_ent = self.get_entity(neighbor_id)
                        if n_ent:
                            nodes.append({**n_ent, "depth": hop})
                    count += 1

            frontier = next_frontier
            if not frontier:
                break

        return {
            "root_entity_id": entity_id,
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "node_count": len(nodes),
                "edge_count": len(edges),
                "max_depth_reached": max(visited.values()) if visited else 0,
            },
        }

    def find_paths(
        self, source_id: str, target_id: str,
        max_depth: int = 5, max_paths: int = 5,
        algorithm: str = "shortest",
    ) -> dict:
        """Find paths between two entities."""
        if source_id not in self.graph or target_id not in self.graph:
            return {"paths": [], "path_count": 0}

        paths_result = []
        try:
            if algorithm == "shortest":
                path = nx.shortest_path(
                    self.graph.to_undirected(), source_id, target_id
                )
                paths_result = [self._format_path(path)]
            elif algorithm == "all_simple":
                all_paths = nx.all_simple_paths(
                    self.graph.to_undirected(), source_id, target_id,
                    cutoff=max_depth,
                )
                for i, p in enumerate(all_paths):
                    if i >= max_paths:
                        break
                    paths_result.append(self._format_path(p))
            elif algorithm == "weighted":
                # Weight by inverse confidence
                for u, v, d in self.graph.edges(data=True):
                    self.graph[u][v]["weight"] = 1.0 / (d.get("confidence", 0.5) + 0.01)
                path = nx.shortest_path(
                    self.graph.to_undirected(), source_id, target_id,
                    weight="weight",
                )
                paths_result = [self._format_path(path)]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass

        return {"source_entity_id": source_id,
                "target_entity_id": target_id,
                "paths": paths_result, "path_count": len(paths_result)}

    def _format_path(self, node_ids: list) -> dict:
        """Convert a list of node IDs into a structured path object."""
        nodes = []
        edges = []
        for nid in node_ids:
            ent = self.get_entity(nid)
            if ent:
                nodes.append({"entity_id": nid, "name": ent.get("name", ""),
                              "type": ent.get("type", "")})

        for i in range(len(node_ids) - 1):
            u, v = node_ids[i], node_ids[i + 1]
            edge_data = None
            if self.graph.has_edge(u, v):
                edge_data = self.graph.edges[u, v, 0]
            elif self.graph.has_edge(v, u):
                edge_data = self.graph.edges[v, u, 0]
            if edge_data:
                edges.append({"type": edge_data.get("relation_type", ""),
                              "confidence": edge_data.get("confidence", 1.0)})

        confidences = [e.get("confidence", 1.0) for e in edges]
        avg_conf = sum(confidences) / len(confidences) if confidences else 1.0
        return {"length": len(node_ids) - 1, "nodes": nodes, "edges": edges,
                "total_confidence": round(avg_conf, 4)}

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self, detailed: bool = False) -> dict:
        """Return graph scale, quality, and distribution statistics."""
        ent_count = self.conn.execute(
            "SELECT COUNT(*) FROM entities WHERE deleted_at IS NULL"
        ).fetchone()[0]
        rel_count = self.conn.execute(
            "SELECT COUNT(*) FROM relations WHERE deleted_at IS NULL"
        ).fetchone()[0]
        doc_count = self.conn.execute(
            "SELECT COUNT(*) FROM documents"
        ).fetchone()[0]
        chunk_count = self.conn.execute(
            "SELECT COUNT(*) FROM chunks"
        ).fetchone()[0]
        avg_conf = self.conn.execute(
            "SELECT AVG(confidence) FROM entities WHERE deleted_at IS NULL"
        ).fetchone()[0] or 0.0

        stats = {
            "scale": {
                "total_entities": ent_count,
                "total_relations": rel_count,
                "total_documents": doc_count,
                "total_chunks": chunk_count,
            },
            "quality": {"avg_confidence": round(avg_conf, 4)},
        }

        if detailed:
            type_dist = {}
            for row in self.conn.execute(
                "SELECT type, COUNT(*) as cnt FROM entities "
                "WHERE deleted_at IS NULL GROUP BY type"
            ):
                type_dist[row["type"]] = row["cnt"]
            rel_dist = {}
            for row in self.conn.execute(
                "SELECT relation_type, COUNT(*) as cnt FROM relations "
                "WHERE deleted_at IS NULL GROUP BY relation_type"
            ):
                rel_dist[row["relation_type"]] = row["cnt"]
            stats["distribution"] = {
                "entities_by_type": type_dist,
                "relations_by_type": rel_dist,
            }

        return stats

    # ------------------------------------------------------------------
    # Entity Resolution (4-stage cascade)
    # ------------------------------------------------------------------

    def _resolve_entity(self, entity: Entity) -> Optional[dict]:
        """Run 4-stage entity resolution. Returns match info or None."""
        # Stage 1: exact name + type match
        row = self.conn.execute(
            "SELECT entity_id, name FROM entities WHERE name=? AND type=? "
            "AND deleted_at IS NULL",
            (entity.name, entity.type),
        ).fetchone()
        if row:
            return {"matched": True, "matched_entity_id": row["entity_id"],
                    "match_stage": "exact", "similarity_score": 1.0}

        # Stage 2: alias match
        for alias in entity.aliases:
            for row in self.conn.execute(
                "SELECT entity_id, aliases FROM entities WHERE type=? "
                "AND deleted_at IS NULL",
                (entity.type,),
            ):
                existing_aliases = json.loads(row["aliases"] or "[]")
                if alias.lower() in [a.lower() for a in existing_aliases]:
                    return {"matched": True,
                            "matched_entity_id": row["entity_id"],
                            "match_stage": "alias",
                            "similarity_score": 1.0}

        # Stage 3: semantic similarity (if embeddings available)
        if self._embeddings and entity.description:
            q_vec = self._get_query_embedding(entity.description)
            if q_vec is not None:
                best_score = 0.0
                best_id = None
                for eid, evec in self._embeddings.items():
                    score = float(
                        np.dot(evec, q_vec) /
                        (np.linalg.norm(evec) * np.linalg.norm(q_vec) + 1e-8)
                    )
                    if score > best_score:
                        best_score = score
                        best_id = eid
                if best_score >= 0.85:
                    return {"matched": True, "matched_entity_id": best_id,
                            "match_stage": "semantic_similarity",
                            "similarity_score": round(best_score, 4)}

        return None

    # ------------------------------------------------------------------
    # Embedding management
    # ------------------------------------------------------------------

    def set_embedding(self, entity_id: str, embedding: list):
        """Store an embedding vector for an entity (in-memory)."""
        self._embeddings[entity_id] = np.array(embedding, dtype=np.float32)

    def _get_query_embedding(self, text: str) -> Optional[np.array]:
        """Get embedding for a query string. Override in production with
        actual embedding model call."""
        # Placeholder: hash-based pseudo-embedding for lightweight mode
        # In production, replace with OpenAI/local model embedding
        if not text:
            return None
        # Simple deterministic pseudo-embedding (replace with real model)
        dim = 384
        vec = np.zeros(dim, dtype=np.float32)
        for i, ch in enumerate(text[:dim]):
            vec[i % dim] += ord(ch) / 1000.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _row_to_entity_dict(self, row) -> dict:
        """Convert a SQLite row to an entity dictionary."""
        return {
            "entity_id": row["entity_id"],
            "name": row["name"],
            "type": row["type"],
            "aliases": json.loads(row["aliases"] or "[]"),
            "attributes": json.loads(row["attributes"] or "{}"),
            "description": row["description"],
            "confidence": row["confidence"],
            "tags": json.loads(row["tags"] or "[]"),
            "provenance": json.loads(row["provenance"] or "{}"),
            "lifecycle_state": row["lifecycle_state"],
            "tenant_id": row["tenant_id"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def save_document(self, doc: Document) -> str:
        """Persist a source document and return its ID."""
        if not doc.doc_id:
            doc.doc_id = _generate_id("doc")
        doc.created_at = _now_iso()
        self.conn.execute(
            "INSERT INTO documents VALUES (?,?,?,?,?,?)",
            (doc.doc_id, doc.title, doc.format, doc.content,
             doc.tenant_id, doc.created_at),
        )
        self.conn.commit()
        return doc.doc_id

    def save_chunk(self, chunk: Chunk) -> str:
        """Persist a text chunk and return its ID."""
        if not chunk.chunk_id:
            chunk.chunk_id = _generate_id("chunk")
        chunk.created_at = _now_iso()
        self.conn.execute(
            "INSERT INTO chunks VALUES (?,?,?,?,?,?)",
            (chunk.chunk_id, chunk.doc_id, chunk.text, chunk.chunk_index,
             json.dumps(chunk.entities_referenced, ensure_ascii=False),
             chunk.created_at),
        )
        self.conn.commit()
        return chunk.chunk_id

    def close(self):
        """Close the database connection."""
        self.conn.close()
