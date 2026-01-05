# nodes Package

LLM-powered nodes for classification and response generation.

**Parent**: @src/enterprise_rag/CLAUDE.md

## classifier.py

See @src/enterprise_rag/nodes/classifier.py

### Purpose

Classifies user queries into intent categories using structured LLM output.

### Key Pattern: Structured Output

```python
structured_llm = llm.with_structured_output(IntentClassification)
classification = structured_llm.invoke(prompt)
```

This uses OpenAI function calling under the hood. The LLM is constrained to return JSON matching `IntentClassification` (defined in @src/enterprise_rag/state.py):

```python
class IntentClassification(TypedDict):
    intent: Literal["internal_docs", "elastic_docs", "jira"]
    reasoning: str
```

### Routing Logic

```python
intent_to_agent = {
    "internal_docs": "internal_docs_agent",
    "elastic_docs": "elastic_docs_agent",
    "jira": "jira_agent",
}
return Command(update={...}, goto=intent_to_agent[intent])
```

### Prompt Structure

The classification prompt includes:

1. Description of each intent category
2. Example queries for each category
3. The data sources each category uses

To add a new intent category:

1. Add to `IntentType` Literal in @src/enterprise_rag/state.py
2. Add description and examples to `CLASSIFICATION_PROMPT` in @src/enterprise_rag/nodes/classifier.py
3. Add routing entry in `intent_to_agent` dict

---

## response.py

See @src/enterprise_rag/nodes/response.py

### Purpose

Generates final response from retrieved chunks with source citations.

### Key Pattern: Context Formatting

Chunks are formatted on-demand inside the node:

```python
for i, chunk in enumerate(chunks, 1):
    source_info = f"[{chunk['source_type']}]"
    if chunk.get("title"):
        source_info += f" {chunk['title']}"
    context_parts.append(f"### Source {i} {source_info}\n{chunk['content']}")
```

This follows the principle: store raw data in state, format prompts inside nodes.

### Source Deduplication

```python
seen_sources: set[tuple[str, str | None]] = set()
for chunk in chunks:
    key = (title, url)
    if key not in seen_sources:
        seen_sources.add(key)
        sources.append(...)
```

Multiple chunks can come from the same source (e.g., different sections of a wiki page). We deduplicate by `(title, url)` tuple.

### Empty Chunks Handling

If no chunks were retrieved, returns a graceful fallback message rather than failing.
