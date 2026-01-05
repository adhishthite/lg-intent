# CLAUDE.md

High-level guidance for Claude Code. See subdirectory CLAUDE.md files for implementation details.

## Documentation Structure

This repo uses **hierarchical CLAUDE.md files**:

- @CLAUDE.md - This file (high-level)
- @src/enterprise_rag/CLAUDE.md - Package details (state, config, graph)
- @src/enterprise_rag/nodes/CLAUDE.md - Node implementations
- @src/enterprise_rag/agents/CLAUDE.md - Agent implementations

**Important**: After making code changes, spin up a subagent to update the relevant CLAUDE.md files. If a subdirectory does not have a CLAUDE.md, create one for that directory. Keep documentation in sync with implementation.

## Project Overview

Enterprise RAG system with intent-based routing. Classifies user queries and routes to specialized retrieval agents.

## Quick Reference

```bash
make install       # Install dependencies
make check         # Format + lint
make test          # Run tests (8 tests)
make run           # Run demo
make graph-ascii   # Print graph
```

## Architecture

```bash
User Query → classify_intent → [agent] → draft_response → Response
                    ↓
         ┌─────────┼─────────┐
         ↓         ↓         ↓
   internal    elastic     jira
     docs       docs
```

**Three intent categories**:

- `internal_docs` - Wiki + ServiceNow
- `elastic_docs` - Elasticsearch documentation
- `jira` - Issue tracker

## Key Files

| File                                        | Purpose                  |
| ------------------------------------------- | ------------------------ |
| @src/enterprise_rag/state.py                | State schemas            |
| @src/enterprise_rag/graph.py                | LangGraph wiring         |
| @src/enterprise_rag/nodes/classifier.py     | Intent classification    |
| @src/enterprise_rag/nodes/response.py       | Response generation      |
| @src/enterprise_rag/agents/internal_docs.py | Internal docs agent      |
| @src/enterprise_rag/agents/elastic_docs.py  | Elasticsearch docs agent |
| @src/enterprise_rag/agents/jira.py          | Jira agent               |

## Environment

Requires Azure OpenAI credentials in `.env`:

- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_KEY`

## Async Implementation

**All code must be async.** This system is designed for FastAPI deployment:

- All nodes use `async def` and `await llm.ainvoke()`
- All agents use `async def` (ready for async HTTP clients like httpx)
- Graph invocation uses `.ainvoke()` or `.astream()`
- Tests use `pytest-asyncio` with `@pytest.mark.asyncio`

When adding new nodes or agents, always use `async def`.

## Extending

To add a new data source, see @src/enterprise_rag/agents/CLAUDE.md for the agent contract and production implementation patterns.
