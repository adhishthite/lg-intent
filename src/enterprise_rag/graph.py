"""
LangGraph workflow for Enterprise RAG.

This module wires together all nodes into a compiled graph.
Uses tool-based architecture where the orchestrator decides which tools to call.
"""

from langgraph.graph import END, START, StateGraph

from enterprise_rag.nodes import (
    draft_response,
    orchestrator,
    reranker,
    should_continue_after_orchestrator,
)
from enterprise_rag.state import RAGState


def create_rag_graph():
    """
    Create and compile the Enterprise RAG graph.

    Graph Structure (Tool-Based):
    =============================

                        START
                          |
                          v
                    orchestrator
                    (LLM + tools)
                          |
              (conditional edge)
                   /          \\
                  v            v
           [has chunks]   [no chunks = general]
                  |            |
                  v            v
              reranker        END
                  |       (general response
                  v        already set)
           draft_response
                  |
                  v
                 END

    Tool Execution:
    ===============
    - Orchestrator LLM decides which tools to call with focused queries
    - Tool calls execute in parallel within orchestrator node
    - Results accumulated in retrieved_chunks
    - Reranker runs cross-source Jina reranking
    - draft_response generates final answer

    General Queries:
    ================
    - If user sends greeting/chitchat, orchestrator responds directly
    - No tools are called, response is set immediately
    - Graph exits without going through reranker/draft_response

    Returns:
        Compiled LangGraph that can be invoked with .ainvoke() or .astream()
    """
    # Create the graph with our state schema
    workflow = StateGraph(RAGState)

    # -------------------------------------------------------------------------
    # ADD NODES
    # -------------------------------------------------------------------------

    # Orchestrator - decides which tools to call, executes them
    workflow.add_node("orchestrator", orchestrator)

    # Cross-source reranking - Jina reranks chunks from all tools
    workflow.add_node("reranker", reranker)

    # Response generation - synthesizes answer from reranked chunks
    workflow.add_node("draft_response", draft_response)

    # -------------------------------------------------------------------------
    # ADD EDGES
    # -------------------------------------------------------------------------

    # Entry point
    workflow.add_edge(START, "orchestrator")

    # Conditional edge after orchestrator
    # - If response is set (general query): go to END
    # - If retrieved_chunks has content: go to reranker
    workflow.add_conditional_edges(
        "orchestrator",
        should_continue_after_orchestrator,
        {
            "reranker": "reranker",
            "__end__": END,
        },
    )

    # Reranker to response generation
    workflow.add_edge("reranker", "draft_response")

    # Exit point
    workflow.add_edge("draft_response", END)

    # -------------------------------------------------------------------------
    # COMPILE
    # -------------------------------------------------------------------------

    return workflow.compile()


def print_graph_ascii():
    """Print ASCII representation of the graph."""
    graph = create_rag_graph()
    print(graph.get_graph().draw_ascii())


def print_graph_mermaid():
    """Print Mermaid diagram of the graph."""
    graph = create_rag_graph()
    print(graph.get_graph().draw_mermaid())
