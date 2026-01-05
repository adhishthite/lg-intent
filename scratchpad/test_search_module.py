"""Test the new search module."""

import asyncio
import sys

sys.path.insert(0, str(__file__).rsplit("/", 2)[0] + "/src")

from enterprise_rag.config import settings
from enterprise_rag.search import close_clients, hybrid_search


async def main():
    try:
        # Test wiki search
        print("=== WIKI SEARCH ===")
        wiki_results = await hybrid_search(
            index=settings.elasticsearch.WIKI_ES_VECTOR_INDEX,
            query="How do I submit a PTO request?",
            source_type="wiki",
            k=3,
        )
        for i, chunk in enumerate(wiki_results, 1):
            print(f"\n{i}. {chunk['title']}")
            print(f"   URL: {chunk['source_url']}")
            print(f"   Score: {chunk['relevance_score']}")

        # Test docs search
        print("\n\n=== DOCS SEARCH ===")
        docs_results = await hybrid_search(
            index=settings.elasticsearch.DOCS_ES_VECTOR_INDEX,
            query="How do I create an Elasticsearch index?",
            source_type="elastic_docs",
            k=3,
        )
        for i, chunk in enumerate(docs_results, 1):
            print(f"\n{i}. {chunk['title']}")
            print(f"   URL: {chunk['source_url']}")
            print(f"   Score: {chunk['relevance_score']}")

    finally:
        await close_clients()


if __name__ == "__main__":
    asyncio.run(main())
