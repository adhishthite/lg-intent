"""
Elasticsearch hybrid search utilities.

Uses the retriever API (ES 8.14+) for BM25 + vector search with RRF fusion.
"""

from elasticsearch import AsyncElasticsearch
from langchain_openai import OpenAIEmbeddings

from enterprise_rag.config import settings
from enterprise_rag.state import RetrievedChunk, SourceType

# Lazy-initialized clients (created on first use)
_es_client: AsyncElasticsearch | None = None
_embeddings: OpenAIEmbeddings | None = None


async def close_clients() -> None:
    """Close ES client. Call on shutdown."""
    global _es_client, _embeddings
    if _es_client is not None:
        await _es_client.close()
        _es_client = None
    # OpenAIEmbeddings doesn't need explicit close
    _embeddings = None


def get_es_client() -> AsyncElasticsearch:
    """Get or create the Elasticsearch async client."""
    global _es_client
    if _es_client is None:
        _es_client = AsyncElasticsearch(
            hosts=[settings.elasticsearch.ELASTICSEARCH_URL],
            api_key=settings.elasticsearch.ELASTICSEARCH_API_KEY,
        )
    return _es_client


def get_embeddings() -> OpenAIEmbeddings:
    """
    Get or create the embeddings client.

    Uses Azure OpenAI via v1 API pattern (same as ChatOpenAI).
    """
    global _embeddings
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(
            model=settings.azure.EMBEDDING_DEPLOYMENT_NAME,
            base_url=settings.azure.OPENAI_ENDPOINT.rstrip("/") + "/openai/v1/",
            api_key=settings.azure.OPENAI_API_KEY,
        )
    return _embeddings


async def get_embedding(text: str) -> list[float]:
    """Get embedding for text using Azure OpenAI via LangChain."""
    embeddings = get_embeddings()
    return await embeddings.aembed_query(text)


async def hybrid_search(
    index: str,
    query: str,
    source_type: SourceType,
    k: int | None = None,
) -> list[RetrievedChunk]:
    """
    Perform hybrid search using BM25 + vector with RRF fusion.

    Uses the retriever API (ES 8.14+) for composing search strategies.

    Args:
        index: Elasticsearch index name
        query: User's search query
        source_type: Source type for RetrievedChunk (e.g., "wiki", "elastic_docs")
        k: Number of results (defaults to settings.elasticsearch.ES_K)

    Returns:
        List of RetrievedChunk objects
    """
    if k is None:
        k = settings.elasticsearch.ES_K

    es = get_es_client()
    query_embedding = await get_embedding(query)

    # Use retriever API for hybrid search with RRF
    result = await es.search(
        index=index,
        size=k,
        retriever={
            "rrf": {
                "retrievers": [
                    # BM25 text search across content and metadata
                    {
                        "standard": {
                            "query": {
                                "bool": {
                                    "should": [
                                        {"match": {"page_content": query}},
                                        {"match": {"metadata.title": {"query": query, "boost": 2}}},
                                        {
                                            "match": {
                                                "metadata.keywords": {"query": query, "boost": 1.5}
                                            }
                                        },
                                        {"match": {"metadata.document_summary": query}},
                                    ]
                                }
                            }
                        }
                    },
                    # Vector search
                    {
                        "knn": {
                            "field": "embedding",
                            "query_vector": query_embedding,
                            "k": k,
                            "num_candidates": 50,
                        }
                    },
                ],
                "rank_constant": settings.elasticsearch.ES_RANK_CONSTANT,
            }
        },
        _source=["page_content", "metadata", "page_id", "chunk_id"],
    )

    # Convert hits to RetrievedChunk format
    chunks: list[RetrievedChunk] = []
    for hit in result["hits"]["hits"]:
        source = hit["_source"]
        metadata = source.get("metadata", {})

        chunks.append(
            {
                "content": source.get("page_content", ""),
                "source_type": source_type,
                "source_url": metadata.get("url"),
                "title": metadata.get("title"),
                "relevance_score": hit.get("_score"),
            }
        )

    return chunks
