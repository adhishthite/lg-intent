"""
Explore Elasticsearch indices to understand schema and data structure.

Focuses on wiki and docs indices for the internal_docs and elastic_docs agents.
"""

import asyncio

# Add parent to path so we can import config
import sys

from elasticsearch import AsyncElasticsearch

sys.path.insert(0, str(__file__).rsplit("/", 2)[0] + "/src")

from enterprise_rag.config import settings


async def explore_indices():
    """Connect to ES and explore wiki + docs indices."""

    # Create async client
    es = AsyncElasticsearch(
        hosts=[settings.elasticsearch.ELASTICSEARCH_URL],
        api_key=settings.elasticsearch.ELASTICSEARCH_API_KEY,
    )

    indices = {
        "wiki": settings.elasticsearch.WIKI_ES_VECTOR_INDEX,
        "docs": settings.elasticsearch.DOCS_ES_VECTOR_INDEX,
    }

    try:
        # Test connection
        info = await es.info()
        print(f"Connected to Elasticsearch {info['version']['number']}")
        print("=" * 60)

        for name, index in indices.items():
            print(f"\n{'=' * 60}")
            print(f"INDEX: {name} ({index})")
            print("=" * 60)

            # Check if index exists
            exists = await es.indices.exists(index=index)
            if not exists:
                print("  Index does not exist!")
                continue

            # Get mapping
            print("\n--- MAPPING ---")
            mapping = await es.indices.get_mapping(index=index)
            properties = mapping[index]["mappings"].get("properties", {})

            for field, config in sorted(properties.items()):
                field_type = config.get("type", "object")
                if field_type == "dense_vector":
                    dims = config.get("dims", "?")
                    print(f"  {field}: {field_type} (dims={dims})")
                elif field_type == "object" or "properties" in config:
                    print(f"  {field}: {field_type}")
                    nested = config.get("properties", {})
                    for nf, nc in sorted(nested.items()):
                        print(f"    .{nf}: {nc.get('type', 'object')}")
                else:
                    print(f"  {field}: {field_type}")

            # Get doc count
            count = await es.count(index=index)
            print(f"\n--- DOCUMENT COUNT: {count['count']} ---")

            # Get sample document
            print("\n--- SAMPLE DOCUMENT ---")
            result = await es.search(
                index=index,
                size=1,
                query={"match_all": {}},
            )

            if result["hits"]["hits"]:
                doc = result["hits"]["hits"][0]["_source"]
                # Print non-vector fields (vectors are too long)
                for key, value in doc.items():
                    if isinstance(value, list) and len(value) > 10:
                        print(f"  {key}: [vector with {len(value)} dims]")
                    elif isinstance(value, str) and len(value) > 200:
                        print(f"  {key}: {value[:200]}...")
                    else:
                        print(f"  {key}: {value}")

    finally:
        await es.close()


if __name__ == "__main__":
    asyncio.run(explore_indices())
