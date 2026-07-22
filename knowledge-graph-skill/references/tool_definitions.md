# Tool Definitions (LLM Function Calling)

All definitions are available at `GET /api/v1/tools` in OpenAI format.

## Tool Selection Guide for LLMs

| User Intent | Recommended Tool |
|-------------|-----------------|
| "Find entities related to X" | search_entities |
| "What does the graph contain?" | get_graph_stats |
| "How are X and Y connected?" | find_paths |
| "Show me X's relationships" | query_subgraph |
| "Answer this complex question" | graphrag_search |
| "Query: specific pattern" | text2cypher |
| "If A->B and B->C, then..." | reason |
| "Add this document's knowledge" | extract_knowledge |
| "Create entity/relation" | create_entity / create_relation |
| "Export as diagram" | export_graph |

## Full JSON Definitions

See `scripts/kg_server.py` -> `TOOL_DEFINITIONS` constant for the complete
list of 10 tool definitions in OpenAI Function Calling format. These can be
copied directly into an LLM agent's tool configuration.

### Minimal Example (Python)

```python
import requests

# Get tool definitions
tools = requests.get("http://localhost:8700/api/v1/tools").json()

# Use with OpenAI client
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What companies did Apple acquire?"}],
    tools=tools,
)
# Execute the tool call returned by the LLM
```
