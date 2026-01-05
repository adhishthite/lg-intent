"""
Response generation node.

Synthesizes a response from retrieved chunks and generates source citations.
"""

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from enterprise_rag.config import settings
from enterprise_rag.state import RAGState, Source

# Initialize the response LLM
_response_llm = ChatOpenAI(
    model=settings.llm.RESPONSE_MODEL,
    temperature=settings.llm.RESPONSE_TEMPERATURE,
)


# =============================================================================
# Response Generation Prompt
# =============================================================================

RESPONSE_PROMPT = """You are a helpful enterprise assistant. Answer the user's question based on the provided context.

## User Question
{query}

## Retrieved Context
{context}

## Instructions
- Answer the question directly and concisely
- Only use information from the provided context
- If the context doesn't contain enough information, say so
- Be professional and helpful
- Do not make up information not present in the context

## Response"""


def draft_response(state: RAGState) -> dict:
    """
    Generate a response from retrieved chunks.

    This node:
    1. Formats retrieved chunks into context
    2. Generates a response using the LLM
    3. Extracts source citations from the chunks

    Args:
        state: Current RAG state with retrieved_chunks populated

    Returns:
        Dict with response and sources updates
    """
    chunks = state["retrieved_chunks"]

    # Handle case where no chunks were retrieved
    if not chunks:
        return {
            "response": "I couldn't find any relevant information to answer your question. "
            "Please try rephrasing or provide more details.",
            "sources": [],
        }

    # Format chunks into context string
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source_info = f"[{chunk['source_type']}]"
        if chunk.get("title"):
            source_info += f" {chunk['title']}"
        context_parts.append(f"### Source {i} {source_info}\n{chunk['content']}")

    context = "\n\n".join(context_parts)

    # Generate response
    prompt = RESPONSE_PROMPT.format(
        query=state["query"],
        context=context,
    )

    response = _response_llm.invoke([HumanMessage(content=prompt)])

    # Extract sources from chunks (deduplicated by title+url)
    seen_sources: set[tuple[str, str | None]] = set()
    sources: list[Source] = []

    for chunk in chunks:
        title = chunk.get("title") or "Untitled"
        url = chunk.get("source_url")
        key = (title, url)

        if key not in seen_sources:
            seen_sources.add(key)
            sources.append(
                {
                    "title": title,
                    "url": url,
                    "source_type": chunk["source_type"],
                }
            )

    return {
        "response": response.content,
        "sources": sources,
    }
