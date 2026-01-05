"""
Response generation node.

Synthesizes a response from retrieved chunks and generates source citations.
"""

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from enterprise_rag.config import settings
from enterprise_rag.state import RAGState, Source

# Initialize the response LLM (Azure OpenAI via v1 API)
_response_llm = ChatOpenAI(
    model=settings.llm.RESPONSE_MODEL,
    base_url=settings.azure.OPENAI_ENDPOINT.rstrip("/") + "/openai/v1/",
    api_key=settings.azure.OPENAI_API_KEY,
    temperature=settings.llm.RESPONSE_TEMPERATURE,
    reasoning={"effort": settings.llm.RESPONSE_REASONING_EFFORT},
    timeout=settings.llm.TIMEOUT_SECONDS,
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
- Answer the question directly and concisely using markdown formatting
- Only use information from the provided context
- If the context doesn't contain enough information, say so
- Be professional and helpful
- Do not make up information not present in the context
- At the end of your response, add a "## Sources" section with markdown links to the sources you used
- Format each source as: `- [Title](URL)`
- Only include sources that you actually referenced in your answer

## Response"""


async def draft_response(state: RAGState) -> dict:
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

    # Format chunks into context string (include URLs for markdown links)
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        title = chunk.get("title") or "Untitled"
        url = chunk.get("source_url") or "No URL"
        source_header = f"### Source {i}: {title}\n**URL**: {url}\n**Type**: {chunk['source_type']}"
        context_parts.append(f"{source_header}\n\n{chunk['content']}")

    context = "\n\n".join(context_parts)

    # Generate response
    prompt = RESPONSE_PROMPT.format(
        query=state["query"],
        context=context,
    )

    response = await _response_llm.ainvoke([HumanMessage(content=prompt)])

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

    # Handle reasoning models that return content as list
    content = response.content
    if isinstance(content, list):
        content = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block) for block in content
        )

    return {
        "response": content,
        "sources": sources,
    }
