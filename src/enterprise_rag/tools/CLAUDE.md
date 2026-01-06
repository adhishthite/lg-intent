# tools Package

Search tools for the orchestrator.

**Parent**: @src/enterprise_rag/CLAUDE.md

## Overview

Tools accept focused queries (not the full user question) and perform retrieval from their respective data sources. Keywords are extracted via LLM for quality.

## Tool Contract

All tools must:

1. Be async functions decorated with `@tool`
2. Accept a focused query string (not full user question)
3. Return `list[dict]` with RetrievedChunk-compatible fields
4. Extract keywords using `_extract_keywords_llm()` for quality

```python
@tool
async def search_my_source(
    query: Annotated[str, Field(description="Focused query for...")],
) -> list[dict]:
    """Docstring describes when to use this tool."""
    # Extract keywords in parallel with hybrid search
    keyword_task = _extract_keywords_llm(query)
    hybrid_task = hybrid_search(index=..., query=query, source_type=...)

    keywords, hybrid_chunks = await asyncio.gather(keyword_task, hybrid_task)
    keyword_chunks = await keyword_search(index=..., keywords=keywords, source_type=...)

    candidates = merge_and_dedupe_chunks(hybrid_chunks, keyword_chunks)
    return [dict(chunk) for chunk in candidates]
```

## Available Tools

### search_internal_docs

Searches internal wiki for company policies, PTO, onboarding, security guidelines.

- **Index**: `elasticgpt-embeddings-wiki`
- **Source type**: `wiki`
- **Use for**: Company policies, HR, internal processes

### search_elastic_docs

Searches Elasticsearch official documentation.

- **Index**: `elasticgpt-embeddings-docs`
- **Source type**: `elastic_docs`
- **Use for**: Elasticsearch queries, Kibana, Elastic Stack

### search_jira

Searches Jira issue tracker.

- **Source type**: `jira`
- **Use for**: Bug status, feature requests, issue tracking
- **Note**: Currently uses mock data

## Keyword Extraction

Tools extract keywords using a lightweight LLM call:

```python
_keyword_llm = ChatOpenAI(
    model=settings.llm.CLASSIFIER_MODEL,  # Fast model
    reasoning={"effort": "low"},
    ...
).with_structured_output(KeywordExtraction)
```

Keyword extraction runs in parallel with hybrid search to minimize latency impact.

## Adding a New Tool

1. Add tool function in `tools/__init__.py`
2. Include in `ALL_TOOLS` list
3. Update orchestrator system prompt in `nodes/orchestrator.py` to describe the new tool
4. Update this CLAUDE.md with tool details
