# agents Package

Retrieval agents for different data sources. Currently mocked - swap implementations for production.

**Parent**: @src/enterprise_rag/CLAUDE.md

## Agent Contract

All agents must:

1. Be async functions (`async def`)
2. Accept `RAGState` as input (defined in @src/enterprise_rag/state.py)
3. Return `Command[Literal["draft_response"]]`
4. Populate `retrieved_chunks` with `list[RetrievedChunk]`

```python
async def my_agent(state: RAGState) -> Command[Literal["draft_response"]]:
    chunks = await retrieve_from_source(state["query"])  # Use async HTTP clients
    return Command(
        update={"retrieved_chunks": chunks},
        goto="draft_response",
    )
```

**Note**: The system is fully async for FastAPI deployment readiness. Mock functions are sync (no I/O), but production implementations should use async HTTP clients (httpx, aiohttp).

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

Searches internal documentation: wiki + ServiceNow.

### Mock Implementation

Returns 3 chunks: 2 from wiki, 1 from ServiceNow.

### Production Implementation

Replace `_mock_wiki_search` and `_mock_servicenow_search` with async versions:

**Wiki (Confluence example with httpx)**:

```python
import httpx

async def _wiki_search(query: str) -> list[RetrievedChunk]:
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{CONFLUENCE_URL}/wiki/rest/api/search",
            params={"cql": f'text ~ "{query}"'},
            headers={"Authorization": f"Bearer {API_TOKEN}"},
        )
        results = response.json()["results"]
        return [
            {
                "content": r["content"]["body"]["storage"]["value"],
                "source_type": "wiki",
                "source_url": r["_links"]["webui"],
                "title": r["title"],
                "relevance_score": r.get("score"),
            }
            for r in results
        ]
```

**ServiceNow**:

```python
async def _servicenow_search(query: str) -> list[RetrievedChunk]:
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{SNOW_URL}/api/now/table/kb_knowledge",
            params={"sysparm_query": f"short_descriptionLIKE{query}"},
        )
        return [...]
```

---

## elastic_docs.py

See @src/enterprise_rag/agents/elastic_docs.py

Searches Elasticsearch official documentation.

### Mock Implementation

Returns 3 chunks with sample ES documentation content.

### Production Implementation

#### **Option 1: Vector search over indexed docs**

```python
async def _elastic_docs_search(query: str) -> list[RetrievedChunk]:
    # Embed query and search vector index (async)
    embedding = await embed_async(query)
    results = await vector_db.asearch(embedding, collection="elastic_docs")
    return [...]
```

#### **Option 2: Elastic site search API**

```python
import httpx

async def _elastic_docs_search(query: str) -> list[RetrievedChunk]:
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://www.elastic.co/search-api",
            params={"q": query}
        )
        return [...]
```

---

## jira.py

See @src/enterprise_rag/agents/jira.py

Searches Elastic Jira issue tracker.

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

def format_issue(issue: dict) -> str:
    fields = issue["fields"]
    assignee = fields.get("assignee")
    return f"""**{issue['key']}**: {fields['summary']}

**Status**: {fields['status']['name']}
**Assignee**: {assignee['displayName'] if assignee else 'Unassigned'}
**Priority**: {fields['priority']['name']}

**Description**:
{fields.get('description') or 'No description'}
"""
```

---

## base.py

See @src/enterprise_rag/agents/base.py

Defines the `RetrievalAgent` protocol for documentation purposes. Not enforced at runtime since agents are functions, not classes.
