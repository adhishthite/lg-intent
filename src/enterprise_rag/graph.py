"""
LangGraph workflow for Enterprise RAG.

This module wires together all nodes and agents into a compiled graph.
"""

from langgraph.graph import END, START, StateGraph

from enterprise_rag.agents import elastic_docs_agent, internal_docs_agent, jira_agent
from enterprise_rag.nodes import classify_intent, draft_response
from enterprise_rag.state import RAGState


def create_rag_graph():
    """
    Create and compile the Enterprise RAG graph.

    Graph Structure:
    ================

        START
          |
          v
    classify_intent -----> Routes via Command based on intent
          |
          +---> internal_docs_agent (wiki + servicenow)
          |
          +---> elastic_docs_agent (ES docs)
          |
          +---> jira_agent (issue tracker)
          |
          v
    draft_response -----> Generates answer from retrieved chunks
          |
          v
         END

    Routing:
    ========
    - classify_intent uses Command to route to the appropriate agent
    - All agents use Command to route to draft_response
    - Only explicit edges are START->classify_intent and draft_response->END

    Returns:
        Compiled LangGraph that can be invoked with .invoke() or .stream()
    """
    # Create the graph with our state schema
    workflow = StateGraph(RAGState)

    # ─────────────────────────────────────────────────────────────────────────
    # ADD NODES
    # ─────────────────────────────────────────────────────────────────────────

    # Intent classification - determines which agent to use
    workflow.add_node("classify_intent", classify_intent)

    # Retrieval agents - each searches their respective data sources
    workflow.add_node("internal_docs_agent", internal_docs_agent)
    workflow.add_node("elastic_docs_agent", elastic_docs_agent)
    workflow.add_node("jira_agent", jira_agent)

    # Response generation - synthesizes answer from retrieved chunks
    workflow.add_node("draft_response", draft_response)

    # ─────────────────────────────────────────────────────────────────────────
    # ADD EDGES
    # ─────────────────────────────────────────────────────────────────────────
    # Only essential edges - nodes handle conditional routing via Command

    # Entry point
    workflow.add_edge(START, "classify_intent")

    # Exit point
    workflow.add_edge("draft_response", END)

    # NOTE: All other routing is handled by Command objects:
    # - classify_intent routes to one of the three agents
    # - Each agent routes to draft_response

    # ─────────────────────────────────────────────────────────────────────────
    # COMPILE
    # ─────────────────────────────────────────────────────────────────────────

    return workflow.compile()


def print_graph_ascii():
    """Print ASCII representation of the graph."""
    graph = create_rag_graph()
    print(graph.get_graph().draw_ascii())


def print_graph_mermaid():
    """Print Mermaid diagram of the graph."""
    graph = create_rag_graph()
    print(graph.get_graph().draw_mermaid())
