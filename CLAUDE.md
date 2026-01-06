# CLAUDE.md

High-level guidance for Claude Code. See subdirectory CLAUDE.md files for implementation details.

## Documentation Structure

This repo uses **hierarchical CLAUDE.md files**:

- @CLAUDE.md - This file (high-level)
- @src/enterprise_rag/CLAUDE.md - Package details (state, config, graph)
- @src/enterprise_rag/nodes/CLAUDE.md - Node implementations
- @src/enterprise_rag/tools/CLAUDE.md - Tool implementations

**Important**: After making code changes, spin up a subagent to update the relevant CLAUDE.md files. If a subdirectory does not have a CLAUDE.md, create one for that directory. Keep documentation in sync with implementation.

## Project Overview

Enterprise RAG system with tool-based orchestration. An orchestrator LLM decides which tools to call and with what queries, naturally decomposing multi-intent queries into focused tool calls.

## Quick Reference

```bash
make install                  # Install dependencies
make check                    # Format + lint
make test                     # Run tests (9 tests)
make run                      # Run demo queries
make run QUERY="your query"   # Run a custom query
make graph-ascii              # Print graph
```

## Architecture

```text
User Query → orchestrator (LLM + tools) → reranker → draft_response
                    |
         ┌─────────┼─────────┐
         ↓         ↓         ↓
   search_internal  search_elastic  search_jira
      docs             docs           (tools)
         |             |              |
         └─────────────┴──────────────┘
                       ↓
              retrieved_chunks
                       ↓
                   reranker
                       ↓
                draft_response
```

**Tool-based routing**:

- Orchestrator LLM decides which tools to call based on query
- Multi-intent queries decomposed into focused tool calls
- Tools execute in parallel within orchestrator node
- General queries (greetings) handled directly without tools

**Three search tools**:

- `search_internal_docs` - Wiki + ServiceNow
- `search_elastic_docs` - Elasticsearch documentation
- `search_jira` - Issue tracker

**Multi-intent example**: "What's the PTO policy and how do I set up Elasticsearch?" causes orchestrator to call:

- `search_internal_docs("PTO policy")`
- `search_elastic_docs("Elasticsearch setup")`

Each tool gets a focused query, avoiding search pollution.

## Key Files

| File                                        | Purpose                        |
| ------------------------------------------- | ------------------------------ |
| @src/enterprise_rag/state.py                | State schemas                  |
| @src/enterprise_rag/graph.py                | LangGraph wiring               |
| @src/enterprise_rag/search.py               | ES hybrid search + Jina rerank |
| @src/enterprise_rag/nodes/orchestrator.py   | LLM + tool orchestration       |
| @src/enterprise_rag/nodes/reranker.py       | Cross-source reranking         |
| @src/enterprise_rag/nodes/response.py       | Response generation            |
| `@src/enterprise_rag/tools/__init__.py`     | Search tools                   |

## Environment

Requires credentials in `.env`:

**Azure OpenAI** (embeddings and LLM):

- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_KEY`
- `AZURE_EMBEDDING_DEPLOYMENT_NAME`

**Jina AI** (reranking):

- `JINA_API_KEY`

**Elasticsearch**:

- `ELASTICSEARCH_URL`
- `ELASTICSEARCH_API_KEY`

## Async Implementation

**All code must be async.** This system is designed for FastAPI deployment:

- All nodes use `async def` and `await llm.ainvoke()`
- All tools use `async def`
- Graph invocation uses `.ainvoke()` or `.astream()`
- Tests use `pytest-asyncio` with `@pytest.mark.asyncio`

When adding new nodes or tools, always use `async def`.

## Extending

To add a new data source, see @src/enterprise_rag/tools/CLAUDE.md for the tool contract and implementation patterns.
