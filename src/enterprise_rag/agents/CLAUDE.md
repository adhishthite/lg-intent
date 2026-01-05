# agents Package

Retrieval agents for different data sources.

**Parent**: @src/enterprise_rag/CLAUDE.md

## Agent Contract

All agents must:

1. Be async functions (`async def`)
2. Accept `RAGState` as input (defined in @src/enterprise_rag/state.py)
3. Return `Command[Literal["draft_response"]]`
4. Populate `retrieved_chunks` with `list[RetrievedChunk]`

```python
async def my_agent(state: RAGState) -> Command[Literal["draft_response"]]:
    chunks = await hybrid_search(
        index="my-index",
        query=state["query"],
        source_type="my_source",
    )
    return Command(
        update={"retrieved_chunks": chunks},
        goto="draft_response",
    )
```

## RetrievedChunk Schema

Defined in @src/enterprise_rag/state.py. Every chunk must have these fields:

```python
class RetrievedChunk(TypedDict):
    content: str              # The actual text
    source_type: SourceType   # "wiki", "servicenow", "elastic_docs", "jira"
    source_url: str | None    # Link to original
    title: str | None         # Document title
    relevance_score: float | None  # 0.0-1.0, for sorting
```

---

## internal_docs.py

See @src/enterprise_rag/agents/internal_docs.py

Searches internal wiki documentation using Elasticsearch hybrid search.

### Implementation

Uses `hybrid_search` from @src/enterprise_rag/search.py against the wiki index:

```python
async def internal_docs_agent(state: RAGState) -> Command[Literal["draft_response"]]:
    chunks = await hybrid_search(
        index=settings.elasticsearch.WIKI_ES_VECTOR_INDEX,
        query=state["query"],
        source_type="wiki",
    )
    return Command(
        update={"retrieved_chunks": chunks},
        goto="draft_response",
    )
```

### Index Details

- **Index**: `elasticgpt-embeddings-wiki`
- **Documents**: 7,383 chunks
- **Source**: Internal Confluence wiki
- **Metadata**: `title`, `url`, `space`
- **Search-enhanced**: `keywords`, `document_summary` (used in BM25)

---

## elastic_docs.py

See @src/enterprise_rag/agents/elastic_docs.py

Searches Elasticsearch official documentation using hybrid search.

### Implementation

Uses `hybrid_search` from @src/enterprise_rag/search.py against the docs index:

```python
async def elastic_docs_agent(state: RAGState) -> Command[Literal["draft_response"]]:
    chunks = await hybrid_search(
        index=settings.elasticsearch.DOCS_ES_VECTOR_INDEX,
        query=state["query"],
        source_type="elastic_docs",
    )
    return Command(
        update={"retrieved_chunks": chunks},
        goto="draft_response",
    )
```

### Index Details

- **Index**: `elasticgpt-embeddings-docs`
- **Documents**: 72,187 chunks
- **Source**: elastic.co documentation
- **Metadata**: `title`, `url`, `filename`, `source_type`, `visibility`
- **Search-enhanced**: `keywords`, `document_summary` (used in BM25)

---

## jira.py

See @src/enterprise_rag/agents/jira.py

Searches Elastic Jira issue tracker. Currently returns mock data.

### Mock Implementation

Returns 2 mock issues with status, assignee, and description.

### Production Implementation

```python
import httpx

async def _jira_search(query: str) -> list[RetrievedChunk]:
    jql = f'text ~ "{query}" ORDER BY updated DESC'

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{JIRA_URL}/rest/api/2/search",
            params={"jql": jql, "maxResults": 5},
            headers={"Authorization": f"Bearer {JIRA_TOKEN}"},
        )
        issues = response.json()["issues"]

    return [
        {
            "content": format_issue(issue),
            "source_type": "jira",
            "source_url": f"{JIRA_URL}/browse/{issue['key']}",
            "title": f"{issue['key']}: {issue['fields']['summary']}",
            "relevance_score": None,
        }
        for issue in issues
    ]
```

---

## Adding a New Agent

1. Create `agents/my_agent.py`
2. Use `hybrid_search` if backed by Elasticsearch, or implement custom retrieval
3. Return `Command` with `retrieved_chunks` and `goto="draft_response"`
4. Register in `agents/__init__.py`
5. Add node and routing in @src/enterprise_rag/graph.py
6. Add intent type in @src/enterprise_rag/state.py
7. Update classifier prompt in @src/enterprise_rag/nodes/classifier.py
