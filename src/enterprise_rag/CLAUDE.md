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

# Azure OpenAI (required)
settings.azure.OPENAI_ENDPOINT             # Azure service URL
settings.azure.OPENAI_API_KEY              # Azure API key

# LLM settings
settings.llm.CLASSIFIER_MODEL              # "gpt-5-nano"
settings.llm.CLASSIFIER_REASONING_EFFORT   # "low"

# Other nested access
settings.elasticsearch.ELASTICSEARCH_URL
settings.langsmith.PROJECT
```

### Key Design Decisions

1. **`load_dotenv()` first** - Sets env vars for external SDKs (OpenAI, LangChain)
2. **Nested `BaseSettings`** - Each group has its own `env_prefix` for automatic env var mapping
3. **No duplication** - Config lives only in nested groups, not duplicated at top level

### Nested Config Groups

| Group           | Env Prefix   | Example Access                             |
| --------------- | ------------ | ------------------------------------------ |
| `llm`           | -            | `settings.llm.CLASSIFIER_MODEL`            |
| `elasticsearch` | -            | `settings.elasticsearch.ELASTICSEARCH_URL` |
| `azure`         | `AZURE_`     | `settings.azure.OPENAI_ENDPOINT`           |
| `langsmith`     | `LANGSMITH_` | `settings.langsmith.PROJECT`               |
| `postgres`      | `POSTGRES_`  | `settings.postgres.DB_URI`                 |
| `eval`          | `EVAL_`      | `settings.eval.AGENT_MODEL`                |
| `retrieval`     | -            | `settings.retrieval.MAX_CHUNKS_PER_AGENT`  |

### LLM Settings

| Setting                       | Default      | Description                       |
| ----------------------------- | ------------ | --------------------------------- |
| `CLASSIFIER_MODEL`            | `gpt-5-nano` | Model for intent classification   |
| `CLASSIFIER_REASONING_EFFORT` | `low`        | Reasoning effort for classifier   |
| `RESPONSE_MODEL`              | `gpt-5-nano` | Model for response generation     |
| `RESPONSE_REASONING_EFFORT`   | `medium`     | Reasoning effort for responses    |
| `TIMEOUT_SECONDS`             | `90`         | Request timeout for all LLM calls |

See @src/enterprise_rag/config.py for all available settings.

## Azure OpenAI Integration

LLMs use Azure OpenAI via the v1 API pattern:

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model=settings.llm.CLASSIFIER_MODEL,
    base_url=settings.azure.OPENAI_ENDPOINT.rstrip("/") + "/openai/v1/",
    api_key=settings.azure.OPENAI_API_KEY,
    reasoning={"effort": settings.llm.CLASSIFIER_REASONING_EFFORT},
    timeout=settings.llm.TIMEOUT_SECONDS,
)
```

**Key pattern**: Appending `/openai/v1/` to Azure endpoint enables full `ChatOpenAI` compatibility, including reasoning models.
