# agents Package

Retrieval agents for different data sources. Currently mocked - swap implementations for production.

**Parent**: @src/enterprise_rag/CLAUDE.md

## Agent Contract

All agents must:

1. Accept `RAGState` as input (defined in @src/enterprise_rag/state.py)
2. Return `Command[Literal["draft_response"]]`
3. Populate `retrieved_chunks` with `list[RetrievedChunk]`

```python
def my_agent(state: RAGState) -> Command[Literal["draft_response"]]:
    chunks = retrieve_from_source(state["query"])
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

Searches internal documentation: wiki + ServiceNow.

### Mock Implementation

Returns 3 chunks: 2 from wiki, 1 from ServiceNow.

### Production Implementation

Replace `_mock_wiki_search` and `_mock_servicenow_search` with:

**Wiki (Confluence example)**:

```python
def _wiki_search(query: str) -> list[RetrievedChunk]:
    # Use Confluence REST API or vector search
    results = confluence_client.search(query)
    return [
        {
            "content": r.body,
            "source_type": "wiki",
            "source_url": r.url,
            "title": r.title,
            "relevance_score": r.score,
        }
        for r in results
    ]
```

**ServiceNow**:

```python
def _servicenow_search(query: str) -> list[RetrievedChunk]:
    # Use ServiceNow Table API or Knowledge API
    results = snow_client.search_kb(query)
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
def _elastic_docs_search(query: str) -> list[RetrievedChunk]:
    # Embed query and search vector index
    embedding = embed(query)
    results = vector_db.search(embedding, collection="elastic_docs")
    return [...]
```

#### **Option 2: Elastic site search API**

```python
def _elastic_docs_search(query: str) -> list[RetrievedChunk]:
    # Use Elastic's own search API
    response = requests.get(
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
def _jira_search(query: str) -> list[RetrievedChunk]:
    # Use Jira REST API with JQL
    jql = f'text ~ "{query}" ORDER BY updated DESC'
    issues = jira_client.search_issues(jql, maxResults=5)

    return [
        {
            "content": format_issue(issue),  # Format as markdown
            "source_type": "jira",
            "source_url": f"https://jira.elastic.co/browse/{issue.key}",
            "title": f"{issue.key}: {issue.fields.summary}",
            "relevance_score": None,  # Jira doesn't provide scores
        }
        for issue in issues
    ]

def format_issue(issue) -> str:
    return f"""**{issue.key}**: {issue.fields.summary}

**Status**: {issue.fields.status.name}
**Assignee**: {issue.fields.assignee.displayName if issue.fields.assignee else 'Unassigned'}
**Priority**: {issue.fields.priority.name}

**Description**:
{issue.fields.description or 'No description'}
"""
```

---

## base.py

See @src/enterprise_rag/agents/base.py

Defines the `RetrievalAgent` protocol for documentation purposes. Not enforced at runtime since agents are functions, not classes.
