# nodes Package

LLM-powered nodes for classification and response generation.

**Parent**: @src/enterprise_rag/CLAUDE.md

## classifier.py

See @src/enterprise_rag/nodes/classifier.py

### Purpose

Classifies user queries into intent categories using structured LLM output.

### LLM Configuration

Uses Azure OpenAI via v1 API with reasoning:

```python
_classifier_llm = ChatOpenAI(
    model=settings.llm.CLASSIFIER_MODEL,  # gpt-5-nano
    base_url=settings.azure.OPENAI_ENDPOINT.rstrip("/") + "/openai/v1/",
    api_key=settings.azure.OPENAI_API_KEY,
    reasoning={"effort": settings.llm.CLASSIFIER_REASONING_EFFORT},  # "low"
    timeout=settings.llm.TIMEOUT_SECONDS,  # 90s
)
```

### Intent Weights

Categories have weights to bias classification when ambiguous:

| Intent          | Weight | Description                                    |
| --------------- | ------ | ---------------------------------------------- |
| `internal_docs` | 1.0    | Default preference                             |
| `elastic_docs`  | 1.0    | Default preference                             |
| `jira`          | 0.5    | Only when explicitly asking about tickets/bugs |

Jira has lower weight because users rarely check issue status via this system.

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

### LLM Configuration

Uses Azure OpenAI via v1 API with reasoning:

```python
_response_llm = ChatOpenAI(
    model=settings.llm.RESPONSE_MODEL,  # gpt-5-nano
    base_url=settings.azure.OPENAI_ENDPOINT.rstrip("/") + "/openai/v1/",
    api_key=settings.azure.OPENAI_API_KEY,
    reasoning={"effort": settings.llm.RESPONSE_REASONING_EFFORT},  # "medium"
    timeout=settings.llm.TIMEOUT_SECONDS,  # 90s
)
```

### Reasoning Model Content Handling

Reasoning models return `content` as a list of blocks instead of a string:

```python
content = response.content
if isinstance(content, list):
    content = "".join(
        block.get("text", "") if isinstance(block, dict) else str(block)
        for block in content
    )
```

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
