"""
Test Jina reranking with mock Elasticsearch data.

This script bypasses ES and tests the reranking pipeline directly.
"""

import asyncio

from enterprise_rag.search import close_clients, rerank_chunks
from enterprise_rag.state import RetrievedChunk


def create_mock_chunks(query: str) -> list[RetrievedChunk]:
    """Create mock chunks that simulate ES retrieval for PTO query."""

    # Simulate what ES might return - mix of relevant and irrelevant
    return [
        {
            "content": "Out of Office (OOO) and Paid Time Off (PTO) Overview. "
            "This page provides general information about leave policies at the company. "
            "For specific procedures, see the sections below.",
            "source_type": "wiki",
            "source_url": "https://wiki.example.com/pto-overview",
            "title": "Out of Office (OOO), Paid Time Off (PTO) and other leave",
            "relevance_score": 72.36,  # High ES score due to title match
        },
        {
            "content": "How to Submit a PTO Request: Step-by-Step Guide. "
            "1. Log into Workday at workday.example.com. "
            "2. Navigate to Time Off > Request Time Off. "
            "3. Select the dates you want to take off. "
            "4. Choose 'PTO' as the time off type. "
            "5. Add any notes for your manager. "
            "6. Click Submit. Your manager will receive a notification.",
            "source_type": "wiki",
            "source_url": "https://wiki.example.com/pto-howto",
            "title": "Submitting Time Off Requests in Workday",
            "relevance_score": 45.2,  # Lower ES score - title doesn't match as well
        },
        {
            "content": "Submit a video to the company video library. "
            "Use the video submission form to upload recordings of presentations, "
            "training sessions, or other company content.",
            "source_type": "wiki",
            "source_url": "https://wiki.example.com/submit-video",
            "title": "Submit a video",
            "relevance_score": 0.048,  # Low score - keyword "submit" matched
        },
        {
            "content": "PTO Approval Workflow. Once you submit a PTO request, "
            "it goes through the following approval chain: "
            "1. Direct manager review (1-2 business days). "
            "2. Auto-approval if manager doesn't respond in 3 days. "
            "3. HR notification for requests over 2 weeks.",
            "source_type": "wiki",
            "source_url": "https://wiki.example.com/pto-approval",
            "title": "Time Off Approval Process",
            "relevance_score": 38.5,
        },
        {
            "content": "Exhibitor Booths - How to create a Leadcapture sync request. "
            "This guide covers setting up lead capture for trade show booths.",
            "source_type": "wiki",
            "source_url": "https://wiki.example.com/leadcapture",
            "title": "Exhibitor Booths - How to create a Leadcapture sync request",
            "relevance_score": 0.045,  # Low score - keyword "request" matched
        },
        {
            "content": "Emergency PTO: If you need to take unplanned time off, "
            "you can submit an emergency PTO request retroactively. "
            "Go to Workday > Time Off > Request Time Off, select past dates, "
            "and choose 'Emergency PTO' as the type. Notify your manager ASAP.",
            "source_type": "wiki",
            "source_url": "https://wiki.example.com/emergency-pto",
            "title": "Emergency and Unplanned Time Off",
            "relevance_score": 42.1,
        },
        {
            "content": "Company Holiday Calendar 2024. The company observes the following "
            "paid holidays: New Year's Day, MLK Day, Presidents Day...",
            "source_type": "wiki",
            "source_url": "https://wiki.example.com/holidays",
            "title": "2024 Holiday Calendar",
            "relevance_score": 15.3,
        },
        {
            "content": "PTO Balance and Accrual. You accrue PTO at a rate of 1.5 days per month. "
            "Check your current balance in Workday > Time Off > Time Off Balance.",
            "source_type": "wiki",
            "source_url": "https://wiki.example.com/pto-balance",
            "title": "Understanding Your PTO Balance",
            "relevance_score": 35.8,
        },
    ]


async def test_reranking():
    """Test the reranking pipeline with mock data."""
    query = "How do I submit a PTO request?"

    print(f"Query: {query}")
    print("=" * 70)

    # Get mock chunks (simulating ES retrieval)
    mock_chunks = create_mock_chunks(query)

    print("\nBEFORE RERANKING (ES scores):")
    print("-" * 70)
    for i, chunk in enumerate(
        sorted(mock_chunks, key=lambda x: x["relevance_score"] or 0, reverse=True), 1
    ):
        print(f"{i}. [{chunk['relevance_score']:.2f}] {chunk['title']}")

    # Rerank with Jina
    print(f"\nReranking {len(mock_chunks)} chunks with Jina...")
    reranked = await rerank_chunks(query, mock_chunks)

    print("\nAFTER RERANKING (Jina scores):")
    print("-" * 70)
    for i, chunk in enumerate(reranked, 1):
        print(f"{i}. [{chunk['relevance_score']:.4f}] {chunk['title']}")
        # Show snippet of content
        content_preview = (
            chunk["content"][:100] + "..." if len(chunk["content"]) > 100 else chunk["content"]
        )
        print(f"   {content_preview}")
        print()

    await close_clients()


if __name__ == "__main__":
    asyncio.run(test_reranking())
