"""
KG Skill Basic Tests - Smoke tests for core functionality.

Run: python scripts/test_basic.py
Or:  pytest scripts/test_basic.py

Tests:
1. Entity CRUD (create, read, update, delete)
2. Relation creation and query
3. PII detection and masking
4. Knowledge extraction pipeline
5. Graph export (Mermaid, text-tree)
"""

import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_entity_crud():
    """Test entity create, read, update, delete lifecycle."""
    from kg_core import KGStore, Entity

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        store = KGStore(db_path=db_path)

        # Create
        entity = Entity(name="TestCompany", type="Organization",
                        description="A test company", confidence=0.9)
        result = store.create_entity(entity)
        assert result["status"] == "created", f"Expected created, got {result['status']}"
        eid = result["entity_id"]

        # Read
        ent = store.get_entity(eid)
        assert ent is not None, "Entity not found after create"
        assert ent["name"] == "TestCompany"

        # Update
        store.update_entity(eid, description="Updated description",
                            attributes={"founded": "2020"})
        ent = store.get_entity(eid)
        assert ent["description"] == "Updated description"
        assert ent["attributes"]["founded"] == "2020"

        # Delete
        store.delete_entity(eid)
        ent = store.get_entity(eid)
        assert ent is None, "Entity should be deleted"

        store.close()
        print("[PASS] test_entity_crud")
    finally:
        os.unlink(db_path)


def test_entity_dedup():
    """Test entity deduplication on create."""
    from kg_core import KGStore, Entity

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        store = KGStore(db_path=db_path)

        # First entity
        e1 = Entity(name="Apple", type="Organization",
                    description="Tech company", confidence=0.9)
        r1 = store.create_entity(e1)
        assert r1["status"] == "created"

        # Duplicate (same name + type)
        e2 = Entity(name="Apple", type="Organization",
                    description="Apple Inc.", confidence=0.85)
        r2 = store.create_entity(e2)
        assert r2["status"] == "merged", f"Expected merged, got {r2['status']}"

        store.close()
        print("[PASS] test_entity_dedup")
    finally:
        os.unlink(db_path)


def test_relation_and_query():
    """Test relation creation and graph query."""
    from kg_core import KGStore, Entity, Relation

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        store = KGStore(db_path=db_path)

        e1 = store.create_entity(Entity(name="Apple", type="Organization"))
        e2 = store.create_entity(Entity(name="Beats", type="Organization"))
        id1, id2 = e1["entity_id"], e2["entity_id"]

        # Create relation
        rel = store.create_relation(Relation(
            source_entity_id=id1, relation_type="ACQUIRED",
            target_entity_id=id2, confidence=0.95,
        ))
        assert rel["status"] == "created"

        # Query subgraph
        sub = store.query_subgraph(id1, depth=2)
        assert sub["stats"]["node_count"] >= 2
        assert sub["stats"]["edge_count"] >= 1

        # Find paths
        paths = store.find_paths(id1, id2)
        assert paths["path_count"] >= 1

        store.close()
        print("[PASS] test_relation_and_query")
    finally:
        os.unlink(db_path)


def test_pii_detection():
    """Test PII detection and masking."""
    from kg_pii import PIIDetector

    detector = PIIDetector()
    text = "Contact john@example.com or call 13812345678 for details."
    masked, report = detector.detect_and_mask(text)

    assert "john@example.com" not in masked, "Email not masked"
    assert "13812345678" not in masked, "Phone not masked"
    assert len(report) == 2, f"Expected 2 PII matches, got {len(report)}"
    assert report[0]["type"] in ("email", "phone_cn")

    print("[PASS] test_pii_detection")


def test_extraction_pipeline():
    """Test the extraction pipeline with rule-based extraction."""
    from kg_core import KGStore
    from kg_extract import ExtractionPipeline

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        store = KGStore(db_path=db_path)
        pipeline = ExtractionPipeline(store, config={
            "security": {"pii_detection": False},
        })

        result = pipeline.extract(
            "Apple was founded by Steve Jobs. Apple acquired Beats Electronics.",
            fmt="text",
            strategy="rule_first",
        )

        assert result["status"] == "completed"
        assert result["summary"]["entities_extracted"] > 0
        assert result["summary"]["relations_extracted"] > 0

        store.close()
        print("[PASS] test_extraction_pipeline")
    finally:
        os.unlink(db_path)


def test_export():
    """Test graph export to Mermaid and text-tree formats."""
    from kg_core import KGStore, Entity, Relation
    from kg_export import GraphExporter

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        store = KGStore(db_path=db_path)
        e1 = store.create_entity(Entity(name="NodeA", type="Concept"))
        e2 = store.create_entity(Entity(name="NodeB", type="Concept"))
        store.create_relation(Relation(
            source_entity_id=e1["entity_id"],
            relation_type="RELATED_TO",
            target_entity_id=e2["entity_id"],
        ))

        exporter = GraphExporter(store)

        # Mermaid
        result = exporter.export("mermaid", entity_ids=[e1["entity_id"]], depth=2)
        assert "graph TD" in result["content"]
        assert result["node_count"] >= 2

        # Text tree
        result = exporter.export("text_tree", entity_ids=[e1["entity_id"]], depth=2)
        assert "NodeA" in result["content"]

        store.close()
        print("[PASS] test_export")
    finally:
        os.unlink(db_path)


def test_stats():
    """Test graph statistics."""
    from kg_core import KGStore, Entity

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        store = KGStore(db_path=db_path)
        store.create_entity(Entity(name="A", type="Organization"))
        store.create_entity(Entity(name="B", type="Person"))

        stats = store.get_stats(detailed=True)
        assert stats["scale"]["total_entities"] >= 2
        assert "distribution" in stats

        store.close()
        print("[PASS] test_stats")
    finally:
        os.unlink(db_path)


if __name__ == "__main__":
    print("Running KG Skill basic tests...\n")
    test_entity_crud()
    test_entity_dedup()
    test_relation_and_query()
    test_pii_detection()
    test_extraction_pipeline()
    test_export()
    test_stats()
    print("\nAll tests passed!")
