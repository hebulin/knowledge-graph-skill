"""
KG Skill Export Module - Graph export utilities.

Supports multiple output formats:
- Mermaid (for markdown embedding)
- JSON-LD (for semantic web interop)
- Text tree (LLM-friendly hierarchical representation)
- CSV (nodes + edges for spreadsheet analysis)
- GraphML (cross-tool compatibility)
"""

import json
import csv
import io
from kg_core import KGStore


class GraphExporter:
    """Export graph data to various formats."""

    def __init__(self, store: KGStore):
        """Initialize with a KGStore instance."""
        self.store = store

    def export(self, format: str, entity_ids: list = None,
               depth: int = 2, include_attributes: bool = True,
               max_nodes: int = 500) -> dict:
        """Export subgraph in the specified format."""
        # Gather subgraph data
        if entity_ids:
            nodes, edges = self._gather_subgraph(entity_ids, depth, max_nodes)
        else:
            nodes, edges = self._gather_all(max_nodes)

        if format == "mermaid":
            content = self._to_mermaid(nodes, edges)
        elif format == "jsonld":
            content = self._to_jsonld(nodes, edges, include_attributes)
        elif format == "text_tree":
            content = self._to_text_tree(nodes, edges)
        elif format == "csv":
            content = self._to_csv(nodes, edges)
        elif format == "graphml":
            content = self._to_graphml(nodes, edges)
        else:
            return {"error": f"Unsupported format: {format}"}

        return {
            "format": format,
            "content": content,
            "node_count": len(nodes),
            "edge_count": len(edges),
        }

    def _gather_subgraph(self, entity_ids: list, depth: int,
                         max_nodes: int) -> tuple:
        """Gather nodes and edges for a subgraph."""
        nodes = []
        edges = []
        visited = set()

        for eid in entity_ids:
            if eid in visited or len(nodes) >= max_nodes:
                continue
            sub = self.store.query_subgraph(eid, depth=depth,
                                            limit_per_hop=20)
            for node in sub.get("nodes", []):
                nid = node.get("entity_id")
                if nid and nid not in visited and len(nodes) < max_nodes:
                    nodes.append(node)
                    visited.add(nid)
            edges.extend(sub.get("edges", []))

        return nodes, edges

    def _gather_all(self, max_nodes: int) -> tuple:
        """Gather all nodes and edges (up to max_nodes)."""
        nodes = []
        for row in self.store.conn.execute(
            "SELECT * FROM entities WHERE deleted_at IS NULL LIMIT ?",
            (max_nodes,),
        ):
            nodes.append(self.store._row_to_entity_dict(row))

        edges = []
        for row in self.store.conn.execute(
            "SELECT * FROM relations WHERE deleted_at IS NULL"
        ):
            edges.append({
                "relation_id": row["relation_id"],
                "source": row["source_entity_id"],
                "type": row["relation_type"],
                "target": row["target_entity_id"],
                "confidence": row["confidence"],
            })
        return nodes, edges

    def _to_mermaid(self, nodes: list, edges: list) -> str:
        """Convert to Mermaid graph syntax."""
        lines = ["graph TD"]
        # Node definitions
        for node in nodes:
            nid = self._sanitize_id(node.get("entity_id", ""))
            name = node.get("name", "Unknown").replace('"', "'")
            ntype = node.get("type", "Other")
            lines.append(f'  {nid}["{name}"]')

        # Edge definitions
        for edge in edges:
            src = self._sanitize_id(edge.get("source", ""))
            tgt = self._sanitize_id(edge.get("target", ""))
            rtype = edge.get("type", "RELATED")
            lines.append(f'  {src} -->|{rtype}| {tgt}')

        return "\n".join(lines)

    def _to_jsonld(self, nodes: list, edges: list,
                   include_attributes: bool) -> str:
        """Convert to JSON-LD format."""
        context = {
            "@vocab": "https://knowledge-graph-skill.org/vocab#",
            "name": "https://schema.org/name",
            "type": "https://schema.org/type",
        }
        graph_nodes = []
        for node in nodes:
            n = {
                "@id": f"urn:kg:{node.get('entity_id', '')}",
                "@type": node.get("type", "Other"),
                "name": node.get("name", ""),
            }
            if include_attributes:
                n["attributes"] = node.get("attributes", {})
                n["description"] = node.get("description", "")
                n["confidence"] = node.get("confidence", 1.0)
            graph_nodes.append(n)

        for edge in edges:
            graph_nodes.append({
                "@id": f"urn:kg:{edge.get('relation_id', '')}",
                "@type": "Relation",
                "source": {"@id": f"urn:kg:{edge.get('source', '')}"},
                "relationType": edge.get("type", ""),
                "target": {"@id": f"urn:kg:{edge.get('target', '')}"},
                "confidence": edge.get("confidence", 1.0),
            })

        return json.dumps({"@context": context, "@graph": graph_nodes},
                          ensure_ascii=False, indent=2)

    def _to_text_tree(self, nodes: list, edges: list) -> str:
        """Convert to text tree format (Markdown-friendly)."""
        if not nodes:
            return "(empty graph)"

        # Build adjacency from edges
        children = {}
        root_ids = {n.get("entity_id") for n in nodes}
        child_ids = set()
        for edge in edges:
            src = edge.get("source")
            tgt = edge.get("target")
            rtype = edge.get("type", "RELATED")
            children.setdefault(src, []).append((rtype, tgt))
            child_ids.add(tgt)

        roots = root_ids - child_ids
        if not roots:
            roots = {nodes[0].get("entity_id")}

        lines = []
        for root_id in roots:
            self._build_tree(root_id, children, nodes, lines, "", True)
        return "\n".join(lines)

    def _build_tree(self, node_id: str, children: dict, nodes: list,
                    lines: list, prefix: str, is_last: bool):
        """Recursively build text tree."""
        node = next((n for n in nodes if n.get("entity_id") == node_id), None)
        if not node:
            return
        name = node.get("name", "Unknown")
        ntype = node.get("type", "")
        connector = "└── " if is_last else "├── "
        lines.append(f"{prefix}{connector}{name} ({ntype})")

        kids = children.get(node_id, [])
        new_prefix = prefix + ("    " if is_last else "│   ")
        for i, (rtype, child_id) in enumerate(kids):
            last = (i == len(kids) - 1)
            child = next((n for n in nodes if n.get("entity_id") == child_id),
                         None)
            child_name = child["name"] if child else child_id
            conn = "└── " if last else "├── "
            lines.append(f"{new_prefix}{conn}[{rtype}] -> {child_name}")
            # Recurse for grand children
            if child_id in children:
                self._build_tree(child_id, children, nodes, lines,
                                 new_prefix + ("    " if last else "│   "),
                                 True)

    def _to_csv(self, nodes: list, edges: list) -> str:
        """Convert to CSV format (nodes and edges sections)."""
        output = io.StringIO()
        writer = csv.writer(output)

        writer.writerow(["# Nodes"])
        writer.writerow(["entity_id", "name", "type", "confidence",
                         "description"])
        for node in nodes:
            writer.writerow([
                node.get("entity_id", ""),
                node.get("name", ""),
                node.get("type", ""),
                node.get("confidence", ""),
                node.get("description", "")[:100],
            ])

        writer.writerow([])
        writer.writerow(["# Edges"])
        writer.writerow(["relation_id", "source", "type", "target",
                         "confidence"])
        for edge in edges:
            writer.writerow([
                edge.get("relation_id", ""),
                edge.get("source", ""),
                edge.get("type", ""),
                edge.get("target", ""),
                edge.get("confidence", ""),
            ])

        return output.getvalue()

    def _to_graphml(self, nodes: list, edges: list) -> str:
        """Convert to GraphML XML format."""
        lines = ['<?xml version="1.0" encoding="UTF-8"?>']
        lines.append('<graphml xmlns="http://graphml.graphdrawing.org/xmlns">')
        lines.append('  <graph edgedirected="true">')

        for node in nodes:
            nid = node.get("entity_id", "")
            name = node.get("name", "").replace("&", "&amp;").replace("<", "&lt;")
            ntype = node.get("type", "")
            lines.append(f'    <node id="{nid}">')
            lines.append(f'      <data key="name">{name}</data>')
            lines.append(f'      <data key="type">{ntype}</data>')
            lines.append('    </node>')

        for edge in edges:
            eid = edge.get("relation_id", "")
            src = edge.get("source", "")
            tgt = edge.get("target", "")
            rtype = edge.get("type", "")
            lines.append(f'    <edge id="{eid}" source="{src}" target="{tgt}">')
            lines.append(f'      <data key="relation_type">{rtype}</data>')
            lines.append('    </edge>')

        lines.append('  </graph>')
        lines.append('</graphml>')
        return "\n".join(lines)

    def _sanitize_id(self, entity_id: str) -> str:
        """Sanitize entity ID for use as Mermaid node identifier."""
        return entity_id.replace("-", "_").replace(":", "_")[:20]
