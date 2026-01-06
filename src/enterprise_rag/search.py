"""
Elasticsearch hybrid search utilities.

Uses the retriever API (ES 8.14+) for BM25 + vector search with RRF fusion.
Includes Jina reranking for improved relevance.
"""

import httpx
from elasticsearch import AsyncElasticsearch
from langchain_openai import OpenAIEmbeddings

from enterprise_rag.config import settings
from enterprise_rag.state import RetrievedChunk, SourceType

# Lazy-initialized clients (created on first use)
_es_client: AsyncElasticsearch | None = None
_embeddings: OpenAIEmbeddings | None = None
_httpx_client: httpx.AsyncClient | None = None


async def close_clients() -> None:
    """Close ES and HTTP clients. Call on shutdown."""
    global _es_client, _embeddings, _httpx_client
    if _es_client is not None:
        await _es_client.close()
        _es_client = None
    if _httpx_client is not None:
        await _httpx_client.aclose()
        _httpx_client = None
    # OpenAIEmbeddings doesn't need explicit close
    _embeddings = None


def get_httpx_client() -> httpx.AsyncClient:
    """Get or create the HTTP client for Jina API calls."""
    global _httpx_client
    if _httpx_client is None:
        _httpx_client = httpx.AsyncClient(timeout=30.0)
    return _httpx_client


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
    Retrieves candidates for reranking - use CANDIDATE_K for broad recall.

    Args:
        index: Elasticsearch index name
        query: User's search query
        source_type: Source type for RetrievedChunk (e.g., "wiki", "elastic_docs")
        k: Number of results (defaults to settings.retrieval.CANDIDATE_K)

    Returns:
        List of RetrievedChunk objects
    """
    if k is None:
        k = settings.retrieval.CANDIDATE_K

    es = get_es_client()
    query_embedding = await get_embedding(query)

    # Use retriever API for hybrid search with RRF
    # page_content is primary, metadata fields are low-weight (for optics, not noise)
    # Jina reranker handles precision, we just need recall
    result = await es.search(
        index=index,
        size=k,
        retriever={
            "rrf": {
                "retrievers": [
                    # BM25 - content-first, metadata low-weight
                    {
                        "standard": {
                            "query": {
                                "bool": {
                                    "should": [
                                        # Primary signal
                                        {"match": {"page_content": query}},
                                        # Low-weight metadata (optics, minimal noise)
                                        {
                                            "match": {
                                                "metadata.title": {"query": query, "boost": 0.5}
                                            }
                                        },
                                        {
                                            "match": {
                                                "metadata.keywords": {"query": query, "boost": 0.2}
                                            }
                                        },
                                        {
                                            "match": {
                                                "metadata.document_summary": {
                                                    "query": query,
                                                    "boost": 0.2,
                                                }
                                            }
                                        },
                                    ]
                                }
                            }
                        }
                    },
                    # Vector search for semantic recall
                    {
                        "knn": {
                            "field": "embedding",
                            "query_vector": query_embedding,
                            "k": k,
                            "num_candidates": k * 2,
                        }
                    },
                ],
                "rank_constant": settings.elasticsearch.ES_RANK_CONSTANT,
                "rank_window_size": k,  # Must be >= size
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


async def keyword_search(
    index: str,
    keywords: list[str],
    source_type: SourceType,
    k: int | None = None,
) -> list[RetrievedChunk]:
    """
    Perform keyword-focused search matching extracted keywords against indexed keywords.

    This search specifically targets the metadata.keywords field where
    curated, LLM-generated keywords are stored.

    Args:
        index: Elasticsearch index name
        keywords: Extracted keywords from user query
        source_type: Source type for RetrievedChunk
        k: Number of results (defaults to settings.retrieval.CANDIDATE_K)

    Returns:
        List of RetrievedChunk objects
    """
    if not keywords:
        return []

    if k is None:
        k = settings.retrieval.CANDIDATE_K

    es = get_es_client()

    # Join keywords for matching against the keywords field
    keywords_query = " ".join(keywords)

    # Content-first, metadata low-weight (matches hybrid_search philosophy)
    result = await es.search(
        index=index,
        size=k,
        query={
            "bool": {
                "should": [
                    # Primary signal
                    {"match": {"page_content": keywords_query}},
                    # Low-weight metadata (optics, minimal noise)
                    {"match": {"metadata.title": {"query": keywords_query, "boost": 0.5}}},
                    {"match": {"metadata.keywords": {"query": keywords_query, "boost": 0.2}}},
                    {
                        "match": {
                            "metadata.document_summary": {"query": keywords_query, "boost": 0.2}
                        }
                    },
                ]
            }
        },
        _source=["page_content", "metadata", "page_id", "chunk_id"],
    )

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


def merge_and_dedupe_chunks(
    *chunk_lists: list[RetrievedChunk],
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    """
    Merge multiple chunk lists, deduplicate by URL, and return top K by score.

    Args:
        *chunk_lists: Variable number of chunk lists to merge
        top_k: Number of top results to return (defaults to CANDIDATE_K)

    Returns:
        Deduplicated and sorted list of chunks
    """
    if top_k is None:
        top_k = settings.retrieval.CANDIDATE_K

    # Deduplicate by URL, keeping highest score
    seen: dict[str | None, RetrievedChunk] = {}
    for chunks in chunk_lists:
        for chunk in chunks:
            url = chunk.get("source_url")
            existing = seen.get(url)
            if existing is None:
                seen[url] = chunk
            else:
                # Keep the one with higher score
                existing_score = existing.get("relevance_score") or 0
                new_score = chunk.get("relevance_score") or 0
                if new_score > existing_score:
                    seen[url] = chunk

    # Sort by score descending and take top K
    merged = list(seen.values())
    merged.sort(key=lambda x: x.get("relevance_score") or 0, reverse=True)

    return merged[:top_k]


async def rerank_chunks(
    query: str,
    chunks: list[RetrievedChunk],
    top_n: int | None = None,
) -> list[RetrievedChunk]:
    """
    Rerank chunks using Jina Reranker API.

    Takes candidate chunks from hybrid search and reorders them
    based on query-document relevance using a cross-encoder model.

    Args:
        query: User's search query
        chunks: Candidate chunks to rerank
        top_n: Number of top results to return (defaults to settings.jina.RERANK_TOP_N)

    Returns:
        Reranked list of chunks (top_n results)
    """
    if not chunks:
        return []

    if top_n is None:
        top_n = settings.jina.RERANK_TOP_N

    # Prepare documents for Jina API
    documents = [chunk["content"] for chunk in chunks]

    client = get_httpx_client()

    response = await client.post(
        "https://api.jina.ai/v1/rerank",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.jina.API_KEY}",
        },
        json={
            "model": settings.jina.RERANK_MODEL,
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "return_documents": False,
        },
    )
    response.raise_for_status()

    result = response.json()

    # Reorder chunks based on Jina scores
    reranked: list[RetrievedChunk] = []
    for item in result["results"]:
        original_index = item["index"]
        jina_score = item["relevance_score"]

        # Copy chunk and update score with Jina's relevance score
        chunk = chunks[original_index].copy()
        chunk["relevance_score"] = jina_score
        reranked.append(chunk)

    return reranked
