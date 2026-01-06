"""
Orchestrator node with tool-calling capabilities.

This node replaces the classifier + router pattern. The LLM decides
which tools to call and with what queries, naturally decomposing
multi-intent queries into focused tool calls.
"""

import asyncio
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from enterprise_rag.config import settings
from enterprise_rag.state import RAGState, RetrievedChunk
from enterprise_rag.tools import ALL_TOOLS

# =============================================================================
# Orchestrator LLM Setup
# =============================================================================

# Initialize orchestrator LLM
_orchestrator_llm = ChatOpenAI(
    model=settings.llm.CLASSIFIER_MODEL,
    base_url=settings.azure.OPENAI_ENDPOINT.rstrip("/") + "/openai/v1/",
    api_key=settings.azure.OPENAI_API_KEY,
    temperature=0.0,  # Deterministic tool selection
    timeout=settings.llm.TIMEOUT_SECONDS,
)

# Bind tools to LLM
_llm_with_tools = _orchestrator_llm.bind_tools(ALL_TOOLS)


# =============================================================================
# System Prompt
# =============================================================================

ORCHESTRATOR_SYSTEM_PROMPT = """You are an enterprise assistant that helps users find information from multiple sources.

You have access to the following search tools:

1. **search_internal_docs** - For company policies, HR, onboarding, security guidelines, internal wiki
2. **search_elastic_docs** - For Elasticsearch, Kibana, and Elastic Stack documentation
3. **search_jira** - For bug reports, feature requests, and issue tracking

## Instructions

1. If the user's question requires information retrieval, call the appropriate tool(s).

2. For multi-topic questions, call MULTIPLE tools with FOCUSED queries for each topic.
   - Example: "What's the PTO policy and how do I create an index?"
   - Call: search_internal_docs(query="PTO policy")
   - Call: search_elastic_docs(query="create an index")

3. If the user is just greeting, thanking, or chatting, respond directly WITHOUT calling any tools.
   - Examples: "Hello!", "Thanks!", "How are you?", "Goodbye"
   - Respond with a friendly, professional message

4. Always decompose complex queries into focused sub-queries for each tool.
   - Each tool call should have a FOCUSED query, not the full user question
   - This ensures each search is relevant and precise

5. You can call multiple tools in parallel - they will execute concurrently.

## Important Guidelines
- NEVER pass the full multi-topic question to a tool
- Each tool query should be 1-10 words, focused on that tool's domain
- For conversational messages, respond directly without tools
- Be professional and helpful in direct responses"""


# =============================================================================
# Orchestrator Node
# =============================================================================


async def orchestrator(state: RAGState) -> dict:
    """
    Orchestrate tool calls based on user query.

    This node:
    1. Sends user query to LLM with bound tools
    2. If LLM returns tool calls, executes them in parallel
    3. If LLM returns text (no tools), handles as general response

    Args:
        state: Current RAG state with query

    Returns:
        State updates with either:
        - retrieved_chunks (for retrieval queries)
        - response (for general/chitchat queries)
    """
    query = state["query"]
    messages = [
        SystemMessage(content=ORCHESTRATOR_SYSTEM_PROMPT),
        HumanMessage(content=query),
    ]

    # First LLM call - may return tool calls or direct response
    response: AIMessage = await _llm_with_tools.ainvoke(messages)

    # If no tool calls, this is a "general" response - handle directly
    if not response.tool_calls:
        content = response.content
        if isinstance(content, list):
            content = "".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            )

        return {
            "response": content,
            "sources": [],
            "final_chunks": [],
            "retrieved_chunks": [],
        }

    # Execute tool calls in parallel
    async def execute_tool_call(tool_call: dict) -> list[dict]:
        """Execute a single tool call and return results."""
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        # Find and execute the tool
        for tool in ALL_TOOLS:
            if tool.name == tool_name:
                results = await tool.ainvoke(tool_args)
                return results

        return []

    # Run all tool calls in parallel
    tasks = [execute_tool_call(tc) for tc in response.tool_calls]
    tool_results = await asyncio.gather(*tasks)

    # Accumulate all retrieved chunks
    all_chunks: list[RetrievedChunk] = []
    for chunks in tool_results:
        for chunk in chunks:
            # Ensure chunk has all required fields
            all_chunks.append(
                {
                    "content": chunk.get("content", ""),
                    "source_type": chunk.get("source_type", "wiki"),
                    "source_url": chunk.get("source_url"),
                    "title": chunk.get("title"),
                    "relevance_score": chunk.get("relevance_score"),
                }
            )

    # Return chunks for reranking
    return {"retrieved_chunks": all_chunks}


# =============================================================================
# Routing Function
# =============================================================================


def should_continue_after_orchestrator(state: RAGState) -> Literal["reranker", "__end__"]:
    """
    Route after orchestrator based on whether tools were called.

    If response is already set (general query), go to END.
    If retrieved_chunks has content, go to reranker.

    Args:
        state: Current RAG state

    Returns:
        Next node name or "__end__"
    """
    if state.get("response") is not None:
        # General response already generated - skip retrieval path
        return "__end__"

    # Has chunks to rerank
    return "reranker"
