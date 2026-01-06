"""
Search tools for the orchestrator.

Each tool accepts a focused query (not the full user query) and performs
retrieval from its respective data source. Keywords are extracted via LLM
for quality (synonyms, expansions).
"""

import asyncio
from typing import Annotated

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import Field
from typing_extensions import TypedDict

from enterprise_rag.config import settings
from enterprise_rag.search import (
    hybrid_search,
    keyword_search,
    merge_and_dedupe_chunks,
)

# =============================================================================
# Keyword Extraction
# =============================================================================


class KeywordExtraction(TypedDict):
    """Structured output for keyword extraction."""

    keywords: list[str]


# Lightweight LLM for keyword extraction
_keyword_llm = ChatOpenAI(
    model=settings.llm.CLASSIFIER_MODEL,
    base_url=settings.azure.OPENAI_ENDPOINT.rstrip("/") + "/openai/v1/",
    api_key=settings.azure.OPENAI_API_KEY,
    temperature=0.0,
    reasoning={"effort": "low"},
    timeout=settings.llm.TIMEOUT_SECONDS,
).with_structured_output(KeywordExtraction)


KEYWORD_EXTRACTION_PROMPT = """Extract 3-6 search keywords from this query.

Guidelines:
- Capture core concepts and entities
- Include domain terms and their expansions (e.g., "PTO" and "paid time off")
- Include synonyms that might appear in documentation
- Use lowercase, concise terms (1-3 words each)

Query: {query}

Extract keywords:"""


async def _extract_keywords_llm(query: str) -> list[str]:
    """
    Extract keywords using LLM for quality (synonyms, expansions).

    Uses lightweight model with structured output for reliable extraction.
    """
    prompt = KEYWORD_EXTRACTION_PROMPT.format(query=query)
    result: KeywordExtraction = await _keyword_llm.ainvoke(prompt)
    return result["keywords"]


# =============================================================================
# Search Tools
# =============================================================================


@tool
async def search_internal_docs(
    query: Annotated[
        str,
        Field(
            description="The focused search query for internal documentation. "
            "Should be specific to company policies, processes, or internal wiki content. "
            "Do NOT pass the full user question - extract the relevant portion."
        ),
    ],
) -> list[dict]:
    """
    Search internal company documentation including wiki and ServiceNow.

    Use this tool for questions about:
    - Company policies (PTO, expenses, travel, onboarding)
    - Internal processes and procedures
    - HR and employee resources
    - Security guidelines and compliance
    - Team documentation and wikis

    Args:
        query: A focused search query (not the full user question, but the relevant portion)

    Returns:
        List of relevant document chunks with content, title, URL, and relevance score
    """
    index = settings.elasticsearch.WIKI_ES_VECTOR_INDEX

    # Run keyword extraction in parallel with hybrid search
    keyword_task = _extract_keywords_llm(query)
    hybrid_task = hybrid_search(index=index, query=query, source_type="wiki")

    keywords, hybrid_chunks = await asyncio.gather(keyword_task, hybrid_task)

    # Run keyword search with extracted keywords
    keyword_chunks = await keyword_search(index=index, keywords=keywords, source_type="wiki")

    # Merge and deduplicate candidates
    candidates = merge_and_dedupe_chunks(hybrid_chunks, keyword_chunks)

    # Return as dicts for tool output serialization
    return [dict(chunk) for chunk in candidates]


@tool
async def search_elastic_docs(
    query: Annotated[
        str,
        Field(
            description="The focused search query for Elasticsearch documentation. "
            "Should be specific to Elasticsearch, Kibana, or Elastic Stack topics. "
            "Do NOT pass the full user question - extract the relevant portion."
        ),
    ],
) -> list[dict]:
    """
    Search Elasticsearch official documentation.

    Use this tool for questions about:
    - Elasticsearch queries (bool, match, term, range, etc.)
    - Index management, mappings, and settings
    - Kibana dashboards and visualizations
    - Elastic Stack configuration (Logstash, Beats)
    - Performance tuning and cluster management
    - Ingest pipelines and data transformation

    Args:
        query: A focused search query (not the full user question, but the relevant portion)

    Returns:
        List of relevant document chunks with content, title, URL, and relevance score
    """
    index = settings.elasticsearch.DOCS_ES_VECTOR_INDEX

    # Run keyword extraction in parallel with hybrid search
    keyword_task = _extract_keywords_llm(query)
    hybrid_task = hybrid_search(index=index, query=query, source_type="elastic_docs")

    keywords, hybrid_chunks = await asyncio.gather(keyword_task, hybrid_task)

    # Run keyword search with extracted keywords
    keyword_chunks = await keyword_search(
        index=index, keywords=keywords, source_type="elastic_docs"
    )

    # Merge and deduplicate candidates
    candidates = merge_and_dedupe_chunks(hybrid_chunks, keyword_chunks)

    return [dict(chunk) for chunk in candidates]


def _mock_jira_search(query: str) -> list[dict]:
    """Mock Jira search - replace with real implementation."""
    return [
        {
            "content": f"**ISSUE-1234**: Search feature bug\n\n"
            f"**Status**: In Progress\n"
            f"**Assignee**: Jane Developer\n"
            f"**Priority**: High\n\n"
            f"**Description**:\n"
            f"Related to query: {query}\n\n"
            f"The search feature is returning incorrect results when "
            f"special characters are used in the query string.\n\n"
            f"**Recent Comments**:\n"
            f"- @jane: Working on a fix, ETA end of week\n"
            f"- @bob: Confirmed reproduction steps",
            "source_type": "jira",
            "source_url": "https://jira.elastic.co/browse/ISSUE-1234",
            "title": "ISSUE-1234: Search feature bug",
            "relevance_score": 0.88,
        },
        {
            "content": "**ISSUE-5678**: Feature request - Advanced filters\n\n"
            "**Status**: Open\n"
            "**Assignee**: Unassigned\n"
            "**Priority**: Medium\n\n"
            "**Description**:\n"
            "Users have requested the ability to save and reuse filter configurations.\n\n"
            "**Acceptance Criteria**:\n"
            "- Users can save current filter state\n"
            "- Saved filters appear in a dropdown\n"
            "- Filters can be shared with team members",
            "source_type": "jira",
            "source_url": "https://jira.elastic.co/browse/ISSUE-5678",
            "title": "ISSUE-5678: Feature request - Advanced filters",
            "relevance_score": 0.72,
        },
    ]


@tool
async def search_jira(
    query: Annotated[
        str,
        Field(
            description="The focused search query for Jira issues. "
            "Should mention specific issue numbers, bug descriptions, or feature requests. "
            "Do NOT pass the full user question - extract the relevant portion."
        ),
    ],
) -> list[dict]:
    """
    Search Jira issue tracker for bugs, features, and issue status.

    Use this tool for questions about:
    - Specific issue status (e.g., "ISSUE-1234", "ES-5678")
    - Bug reports and their current state
    - Feature requests and roadmap items
    - Issue assignments and priorities
    - Release blockers and critical bugs

    Args:
        query: A focused search query about Jira issues

    Returns:
        List of relevant Jira issues with status, assignee, and description
    """
    # Mock is sync, production will use async HTTP
    chunks = _mock_jira_search(query)

    return [dict(chunk) for chunk in chunks]


# =============================================================================
# Export
# =============================================================================

ALL_TOOLS = [search_internal_docs, search_elastic_docs, search_jira]

__all__ = [
    "search_internal_docs",
    "search_elastic_docs",
    "search_jira",
    "ALL_TOOLS",
]
