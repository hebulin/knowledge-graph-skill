# Data Model Reference

## Entity

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| entity_id | string | auto | Unique ID (ent_{ulid}) |
| name | string | yes | Canonical name |
| type | enum | yes | Organization/Person/Product/Event/Location/Concept/Other |
| aliases | string[] | no | Alternative names |
| attributes | object | no | Key-value properties (supports temporal values) |
| description | string | no | Natural language description (for embedding) |
| confidence | float | no | 0-1, default 1.0 |
| tags | string[] | no | Custom tags |
| provenance | object | no | Source tracking (doc_id, chunk_id, method) |
| lifecycle_state | enum | auto | active/decaying/deprecated/archived |
| created_at | datetime | auto | Creation timestamp |
| updated_at | datetime | auto | Last update timestamp |
| deleted_at | datetime | null | Soft-delete timestamp |

## Relation

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| relation_id | string | auto | Unique ID (rel_{ulid}) |
| source_entity_id | string | yes | Source entity |
| relation_type | string | yes | UPPER_SNAKE_CASE (e.g. ACQUIRED) |
| target_entity_id | string | yes | Target entity |
| attributes | object | no | Relation properties |
| confidence | float | no | 0-1, default 1.0 |
| direction | enum | no | directed/undirected |
| valid_from | date | no | Temporal validity start |
| valid_to | date | no | Temporal validity end (null = ongoing) |

## Event (N-ary relation)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| event_id | string | auto | Unique ID (evt_{ulid}) |
| event_type | enum | yes | Acquisition/Merger/Launch/Meeting/Transaction/Other |
| participants | object[] | yes | [{entity_id, role}] (min 2) |
| attributes | object | no | Event properties (amount, currency, etc.) |
| occurred_at | datetime | yes | When the event happened |

## Provenance Object

| Field | Type | Description |
|-------|------|-------------|
| source_doc_id | string | Document ID |
| source_chunk_id | string | Chunk ID |
| extraction_method | enum | rule-based/small-model/llm-assisted/manual |
| extraction_confidence | float | Extraction confidence |
| extracted_at | datetime | When extracted |
| verified_by | string | Verification method |

See `assets/schemas/` for full JSON Schema definitions.
