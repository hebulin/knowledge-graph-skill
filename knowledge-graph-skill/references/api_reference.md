# API Reference

All 14 endpoints. Base URL: `http://localhost:8700/api/v1`

## 1. POST /extract
Extract knowledge from a document.
- Body: `{content, format, extraction_strategy, chunk_size, chunk_overlap, auto_resolve}`
- Returns: `{task_id, status, summary, entities, relations}`

## 2. POST /entities
Create an entity (auto-dedup).
- Body: `{name, type, aliases, attributes, description, tags, confidence}`
- Returns: `{entity_id, status, deduplication}`

## 3. PATCH /entities/{entity_id}
Update entity (partial).
- Body: `{name, attributes, description, tags, confidence, temporal}`
- Returns: `{entity_id, status, updated_fields, previous_values}`

## 4. DELETE /entities/{entity_id}
Soft-delete entity.
- Query: `cascade=true, reason`
- Returns: `{entity_id, status, cascade_deleted_relations, deleted_at}`

## 5. POST /relations
Create relation (validates constraints).
- Body: `{source_entity_id, relation_type, target_entity_id, attributes, confidence, valid_from, valid_to}`
- Returns: `{relation_id, status}` or 422 on constraint violation

## 6. POST /search/entities
Hybrid entity search.
- Body: `{query, entity_types, min_confidence, top_k, search_mode}`
- Returns: `{total, results[{entity_id, name, type, score}]}`

## 7. POST /graph/subgraph
Extract N-hop subgraph.
- Body: `{entity_id, depth, relation_types, direction, limit_per_hop, min_confidence}`
- Returns: `{root_entity_id, nodes[], edges[], stats}`

## 8. POST /query/text2cypher
Natural language to Cypher.
- Body: `{question, context_entities, max_results, dry_run}`
- Returns: `{question, generated_cypher, cypher_valid, results, result_count}`

## 9. POST /graphrag/search
GraphRAG hybrid retrieval.
- Body: `{query, strategy, max_context_entities, max_context_relations, include_source_chunks, subgraph_depth, min_confidence}`
- Returns: `{query, strategy_used, context{entities, relations, community_summaries, source_chunks}, stats}`

## 10. POST /graph/paths
Find paths between entities.
- Body: `{source_entity_id, target_entity_id, max_depth, max_paths, algorithm}`
- Returns: `{source_entity_id, target_entity_id, paths[{length, nodes, edges, total_confidence}], path_count}`

## 11. POST /reason
Neuro-symbolic reasoning.
- Body: `{query, reasoning_type, context_entity_ids, max_inference_depth, verify}`
- Returns: `{query, reasoning_type, inference_chain[], conclusion, overall_confidence}`

## 12. GET /stats
Graph statistics.
- Query: `detailed`
- Returns: `{scale{total_entities, total_relations, ...}, quality{avg_confidence}, distribution}`

## 13. POST /export
Export graph to format.
- Body: `{format, entity_ids, depth, include_attributes, max_nodes}`
- Returns: `{format, content, node_count, edge_count}`

## 14. POST /import
Batch import.
- Body: `{format, data, mode, auto_resolve}`
- Returns: `{status, imported, failed, total}`

## Tool Calling
GET /tools returns OpenAI-compatible function definitions for all endpoints.
