# enterprise_rag Package

Core package for the Enterprise RAG system.

**Parent**: @CLAUDE.md
**Children**: @src/enterprise_rag/nodes/CLAUDE.md | @src/enterprise_rag/agents/CLAUDE.md

## Module Overview

| Module                        | Purpose                           |
| ----------------------------- | --------------------------------- |
| @src/enterprise_rag/state.py  | TypedDict schemas for graph state |
| @src/enterprise_rag/config.py | Environment-based configuration   |
| @src/enterprise_rag/graph.py  | LangGraph StateGraph wiring       |

## State Design

The state schema in @src/enterprise_rag/state.py follows LangGraph best practices:

```python
class RAGState(TypedDict):
    query: str                              # Input - never mutated
    intent: IntentType | None               # Set by classifier
    classification_reasoning: str | None    # Set by classifier
    retrieved_chunks: list[RetrievedChunk]  # Set by agents
    response: str | None                    # Set by draft_response
    sources: list[Source]                   # Set by draft_response
```

**Design principles**:

- Store raw data, not formatted prompts
- Each field has a clear owner (which node sets it)
- Use `| None` for fields set during execution
- Use empty lists `[]` as initial values for list fields

## Graph Wiring

In @src/enterprise_rag/graph.py, only 2 explicit edges are defined:

```python
workflow.add_edge(START, "classify_intent")
workflow.add_edge("draft_response", END)
```

All other routing uses Command-based patterns - nodes return `Command(update={...}, goto="next_node")`.

## Type Aliases

Defined in @src/enterprise_rag/state.py:

```python
IntentType = Literal["internal_docs", "elastic_docs", "jira"]
SourceType = Literal["wiki", "servicenow", "elastic_docs", "jira"]
```

Using `Literal` instead of `Enum` for better LLM structured output compatibility.

## Configuration

Configuration uses `pydantic-settings` with nested `BaseSettings` groups in @src/enterprise_rag/config.py.

### Usage Pattern

```python
from enterprise_rag.config import settings

# Top-level
settings.OPENAI_API_KEY

# Nested access
settings.llm.CLASSIFIER_MODEL              # "gpt-4.1-nano"
settings.elasticsearch.ELASTICSEARCH_URL   # ES cluster URL
settings.elasticsearch.WIKI_ES_VECTOR_INDEX
settings.langsmith.PROJECT
settings.azure.OPENAI_ENDPOINT
```

### Key Design Decisions

1. **`load_dotenv()` first** - Sets env vars for external SDKs (OpenAI, LangChain)
2. **Nested `BaseSettings`** - Each group has its own `env_prefix` for automatic env var mapping
3. **No duplication** - Config lives only in nested groups, not duplicated at top level

### Nested Config Groups

| Group           | Env Prefix     | Example Access                          |
| --------------- | -------------- | --------------------------------------- |
| `llm`           | -              | `settings.llm.CLASSIFIER_MODEL`         |
| `elasticsearch` | -              | `settings.elasticsearch.ELASTICSEARCH_URL` |
| `azure`         | `AZURE_`       | `settings.azure.OPENAI_ENDPOINT`        |
| `langsmith`     | `LANGSMITH_`   | `settings.langsmith.PROJECT`            |
| `postgres`      | `POSTGRES_`    | `settings.postgres.DB_URI`              |
| `eval`          | `EVAL_`        | `settings.eval.AGENT_MODEL`             |
| `retrieval`     | -              | `settings.retrieval.MAX_CHUNKS_PER_AGENT` |

See @src/enterprise_rag/config.py for all available settings.
