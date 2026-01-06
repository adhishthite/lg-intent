# enterprise_rag Package

Core package for the Enterprise RAG system.

**Parent**: @CLAUDE.md
**Children**: @src/enterprise_rag/nodes/CLAUDE.md | @src/enterprise_rag/tools/CLAUDE.md

## Module Overview

| Module                        | Purpose                                  |
| ----------------------------- | ---------------------------------------- |
| @src/enterprise_rag/state.py  | TypedDict schemas for graph state        |
| @src/enterprise_rag/config.py | Environment-based configuration          |
| @src/enterprise_rag/graph.py  | LangGraph StateGraph wiring              |
| @src/enterprise_rag/search.py | Elasticsearch hybrid search (BM25 + kNN) |

## State Design (Tool-Based Architecture)

The state schema in @src/enterprise_rag/state.py is simplified for tool-based orchestration:

```python
class RAGState(TypedDict):
    query: str                              # Input - never mutated
    retrieved_chunks: Annotated[list[RetrievedChunk], operator.add]  # From tools
    final_chunks: list[RetrievedChunk]      # After reranking
    response: str | None                    # Set by orchestrator or draft_response
    sources: list[Source]                   # Set by draft_response
```

**Design principles**:

- Store raw data, not formatted prompts
- Each field has a clear owner (which node sets it)
- Use `| None` for fields set during execution
- Use `operator.add` reducer for parallel tool accumulation

**Note**: Intent classification fields (`intents`, `query_keywords`) were removed. Intent is now implicit in which tools the orchestrator calls.

## Graph Wiring

In @src/enterprise_rag/graph.py:

```python
workflow.add_edge(START, "orchestrator")
workflow.add_conditional_edges(
    "orchestrator",
    should_continue_after_orchestrator,
    {"reranker": "reranker", "__end__": END},
)
workflow.add_edge("reranker", "draft_response")
workflow.add_edge("draft_response", END)
```

Three nodes: `orchestrator` -> `reranker` -> `draft_response`

General queries (greetings) exit directly from orchestrator without going through reranker.

## Type Aliases

Defined in @src/enterprise_rag/state.py:

```python
SourceType = Literal["wiki", "servicenow", "elastic_docs", "jira"]
```

## Configuration

Configuration uses `pydantic-settings` with nested `BaseSettings` groups in @src/enterprise_rag/config.py.

### Usage Pattern

```python
from enterprise_rag.config import settings

# Azure OpenAI (required)
settings.azure.OPENAI_ENDPOINT             # Azure service URL
settings.azure.OPENAI_API_KEY              # Azure API key

# LLM settings
settings.llm.CLASSIFIER_MODEL              # "gpt-5-nano" (used by orchestrator)
settings.llm.RESPONSE_MODEL                # "gpt-5-nano"

# Other nested access
settings.elasticsearch.ELASTICSEARCH_URL
settings.jina.API_KEY
```

### Nested Config Groups

| Group           | Env Prefix   | Example Access                             |
| --------------- | ------------ | ------------------------------------------ |
| `llm`           | -            | `settings.llm.CLASSIFIER_MODEL`            |
| `elasticsearch` | -            | `settings.elasticsearch.ELASTICSEARCH_URL` |
| `azure`         | `AZURE_`     | `settings.azure.OPENAI_ENDPOINT`           |
| `langsmith`     | `LANGSMITH_` | `settings.langsmith.PROJECT`               |
| `retrieval`     | -            | `settings.retrieval.CANDIDATE_K`           |
| `jina`          | `JINA_`      | `settings.jina.API_KEY`                    |

### LLM Settings

| Setting                       | Default      | Description                       |
| ----------------------------- | ------------ | --------------------------------- |
| `CLASSIFIER_MODEL`            | `gpt-5-nano` | Model for orchestrator + keywords |
| `RESPONSE_MODEL`              | `gpt-5-nano` | Model for response generation     |
| `RESPONSE_REASONING_EFFORT`   | `medium`     | Reasoning effort for responses    |
| `TIMEOUT_SECONDS`             | `90`         | Request timeout for all LLM calls |

## Azure OpenAI Integration

LLMs use Azure OpenAI via the v1 API pattern:

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model=settings.llm.CLASSIFIER_MODEL,
    base_url=settings.azure.OPENAI_ENDPOINT.rstrip("/") + "/openai/v1/",
    api_key=settings.azure.OPENAI_API_KEY,
    timeout=settings.llm.TIMEOUT_SECONDS,
)
```

**Key pattern**: Appending `/openai/v1/` to Azure endpoint enables full `ChatOpenAI` compatibility.

## Elasticsearch Hybrid Search

The search module in @src/enterprise_rag/search.py provides hybrid search (BM25 + vector with RRF fusion):

```python
from enterprise_rag.search import hybrid_search, close_clients

# Perform hybrid search
chunks = await hybrid_search(
    index=settings.elasticsearch.WIKI_ES_VECTOR_INDEX,
    query="How do I submit PTO?",
    source_type="wiki",
)

# Clean up on shutdown
await close_clients()
```

### Two-Stage Retrieval Pipeline

Tools use a **retrieve-then-rerank** pattern:

```text
Stage 1: Candidate Generation (15 docs per source)
├── hybrid_search (BM25 + kNN + RRF) ─┐
└── keyword_search (parallel) ────────┴─→ merge_and_dedupe

Stage 2: Reranking (top 8)
└── rerank_chunks (Jina cross-encoder in reranker node)
```

### Index Schema

| Index                        | Docs   | Content Field  | Metadata Fields                           |
| ---------------------------- | ------ | -------------- | ----------------------------------------- |
| `elasticgpt-embeddings-wiki` | 7,383  | `page_content` | `title`, `url`, `space`, `keywords`       |
| `elasticgpt-embeddings-docs` | 72,187 | `page_content` | `title`, `url`, `source_type`, `filename` |

Both indices use 1536-dim embeddings (text-embedding-3-small).

## Async Execution

The system is fully async for FastAPI deployment readiness:

```python
import asyncio
from enterprise_rag import create_rag_graph

async def main():
    graph = create_rag_graph()

    # Async invocation
    result = await graph.ainvoke({"query": "How do I submit PTO?"})

    # Or async streaming
    async for event in graph.astream(initial_state, stream_mode="updates"):
        print(event)

asyncio.run(main())
```

**Key points**:

- All nodes (`orchestrator`, `reranker`, `draft_response`) use `async def`
- All tools use `async def`
- Graph invocation uses `.ainvoke()` or `.astream()`
