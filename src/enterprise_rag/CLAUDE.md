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

Environment variables loaded in @src/enterprise_rag/config.py via `python-dotenv`:

| Variable                 | Default      | Used By                                 |
| ------------------------ | ------------ | --------------------------------------- |
| `CLASSIFIER_MODEL`       | gpt-4.1-nano | @src/enterprise_rag/nodes/classifier.py |
| `RESPONSE_MODEL`         | gpt-4.1-nano | @src/enterprise_rag/nodes/response.py   |
| `CLASSIFIER_TEMPERATURE` | 0.0          | @src/enterprise_rag/nodes/classifier.py |
| `RESPONSE_TEMPERATURE`   | 0.3          | @src/enterprise_rag/nodes/response.py   |
