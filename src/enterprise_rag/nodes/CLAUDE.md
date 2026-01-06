# nodes Package

LLM-powered nodes for orchestration and response generation.

**Parent**: @src/enterprise_rag/CLAUDE.md

## Overview

The tool-based architecture has three nodes:

1. **orchestrator** - Decides which tools to call, executes them
2. **reranker** - Cross-source reranks all tool results with Jina
3. **draft_response** - Generates final response from reranked chunks

---

## orchestrator.py

See @src/enterprise_rag/nodes/orchestrator.py

### Purpose

Orchestrates tool calls based on user query. Uses LLM with bound tools to decide which searches to perform and with what focused queries.

### LLM Configuration

```python
_orchestrator_llm = ChatOpenAI(
    model=settings.llm.CLASSIFIER_MODEL,  # gpt-5-nano
    base_url=settings.azure.OPENAI_ENDPOINT.rstrip("/") + "/openai/v1/",
    api_key=settings.azure.OPENAI_API_KEY,
    temperature=0.0,  # Deterministic tool selection
    timeout=settings.llm.TIMEOUT_SECONDS,
)

_llm_with_tools = _orchestrator_llm.bind_tools(ALL_TOOLS)
```

### Behavior

**For retrieval queries:**

- LLM returns tool calls with focused queries
- Tools execute in parallel via `asyncio.gather`
- Results accumulated in `retrieved_chunks`
- Returns `{"retrieved_chunks": all_chunks}`

**For general queries (greetings, chitchat):**

- LLM responds directly without calling tools
- Returns `{"response": content, "sources": [], "final_chunks": [], "retrieved_chunks": []}`
- Graph exits directly to END

### System Prompt

The orchestrator system prompt instructs the LLM to:

1. Call appropriate tools with focused queries
2. Decompose multi-topic questions into multiple tool calls
3. Respond directly for conversational messages

### Routing Function

```python
def should_continue_after_orchestrator(state) -> Literal["reranker", "__end__"]:
    if state.get("response"):  # General query - already answered
        return "__end__"
    return "reranker"  # Has chunks to rerank
```

---

## reranker.py

See @src/enterprise_rag/nodes/reranker.py

### Purpose

Cross-source reranks chunks from all tool calls using Jina cross-encoder.

### Pipeline

1. Collect all chunks from `retrieved_chunks` (from tool calls)
2. Rerank using Jina cross-encoder (`jina-reranker-v3`)
3. Store top 8 results in `final_chunks`

### Output

```python
return {"final_chunks": reranked}
```

---

## response.py

See @src/enterprise_rag/nodes/response.py

### Purpose

Generates final response from reranked chunks with source citations.

### LLM Configuration

```python
_response_llm = ChatOpenAI(
    model=settings.llm.RESPONSE_MODEL,  # gpt-5-nano
    reasoning={"effort": settings.llm.RESPONSE_REASONING_EFFORT},  # "medium"
    ...
)
```

### Key Pattern: Context Formatting

Chunks are formatted on-demand inside the node:

```python
for i, chunk in enumerate(chunks, 1):
    title = chunk.get("title") or "Untitled"
    url = chunk.get("source_url") or "No URL"
    source_header = f"### Source {i}: {title}\n**URL**: {url}\n**Type**: {chunk['source_type']}"
    context_parts.append(f"{source_header}\n\n{chunk['content']}")
```

### Output

```python
return {
    "response": content,
    "sources": sources,  # Deduplicated by (title, url)
}
```
