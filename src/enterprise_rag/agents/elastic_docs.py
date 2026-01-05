"""
Elasticsearch documentation agent.

Retrieves from Elasticsearch official documentation.
Currently returns mock data - swap with real retrieval later.
"""

from typing import Literal

from langgraph.types import Command

from enterprise_rag.state import RAGState, RetrievedChunk


def _mock_elastic_docs_search(query: str) -> list[RetrievedChunk]:
    """Mock Elasticsearch docs search - replace with real implementation."""
    return [
        {
            "content": f"Elasticsearch documentation for: {query}\n\n"
            "## Overview\n"
            "Elasticsearch is a distributed, RESTful search and analytics engine.\n\n"
            "## Example\n"
            "```json\n"
            "GET /my-index/_search\n"
            "{\n"
            '  "query": {\n'
            '    "match": {\n'
            '      "message": "search term"\n'
            "    }\n"
            "  }\n"
            "}\n"
            "```\n\n"
            "## Key Parameters\n"
            "- `query`: The query DSL definition\n"
            "- `size`: Number of hits to return\n"
            "- `from`: Pagination offset",
            "source_type": "elastic_docs",
            "source_url": "https://www.elastic.co/guide/en/elasticsearch/reference/current/search-search.html",
            "title": "Search API Reference",
            "relevance_score": 0.95,
        },
        {
            "content": "## Best Practices\n\n"
            "When designing your search queries:\n\n"
            "1. **Use filters for exact matches** - Filters are cached and faster\n"
            "2. **Limit result size** - Use pagination for large result sets\n"
            "3. **Use source filtering** - Only request fields you need\n"
            "4. **Consider query complexity** - Simpler queries are faster",
            "source_type": "elastic_docs",
            "source_url": "https://www.elastic.co/guide/en/elasticsearch/reference/current/search-best-practices.html",
            "title": "Search Best Practices",
            "relevance_score": 0.82,
        },
        {
            "content": "## Kibana Integration\n\n"
            "You can visualize your Elasticsearch data using Kibana:\n\n"
            "1. Create an index pattern matching your indices\n"
            "2. Use Discover to explore your data\n"
            "3. Build visualizations and dashboards\n"
            "4. Set up alerts for important conditions",
            "source_type": "elastic_docs",
            "source_url": "https://www.elastic.co/guide/en/kibana/current/introduction.html",
            "title": "Kibana Introduction",
            "relevance_score": 0.71,
        },
    ]


def elastic_docs_agent(state: RAGState) -> Command[Literal["draft_response"]]:
    """
    Retrieve from Elasticsearch official documentation.

    Searches the Elastic documentation and routes to draft_response
    for answer generation.

    Args:
        state: Current RAG state with query

    Returns:
        Command with retrieved_chunks update, routing to draft_response
    """
    query = state["query"]

    chunks = _mock_elastic_docs_search(query)

    return Command(
        update={"retrieved_chunks": chunks},
        goto="draft_response",
    )
