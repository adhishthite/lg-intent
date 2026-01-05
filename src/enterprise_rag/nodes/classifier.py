"""
Intent classification node.

Analyzes user queries and routes them to the appropriate retrieval agent.
Uses structured LLM output for reliable classification.
"""

from typing import Literal

from langchain_openai import ChatOpenAI
from langgraph.types import Command

from enterprise_rag.config import settings
from enterprise_rag.state import IntentClassification, RAGState

# Initialize the classifier LLM
_classifier_llm = ChatOpenAI(
    model=settings.llm.CLASSIFIER_MODEL,
    temperature=settings.llm.CLASSIFIER_TEMPERATURE,
)


# =============================================================================
# Classification Prompt
# =============================================================================

CLASSIFICATION_PROMPT = """You are an intent classifier for an enterprise support system.

Analyze the user's query and classify it into one of three categories:

1. **internal_docs** - Questions about internal company processes, policies, or documentation
   - Examples: "How do I submit a PTO request?", "What's the onboarding process?", "Where do I find the security policy?"
   - Sources: Internal wiki, ServiceNow knowledge base

2. **elastic_docs** - Questions about Elasticsearch, Kibana, or Elastic Stack products
   - Examples: "How do I create an index?", "What's the syntax for a bool query?", "How do I configure Kibana dashboards?"
   - Sources: Elasticsearch official documentation

3. **jira** - Questions about specific bugs, issues, or feature requests
   - Examples: "What's the status of ISSUE-1234?", "Are there any open bugs for the search feature?", "Who's assigned to the login issue?"
   - Sources: Elastic Jira issue tracker

User Query: {query}

Classify this query and provide brief reasoning."""


def classify_intent(
    state: RAGState,
) -> Command[Literal["internal_docs_agent", "elastic_docs_agent", "jira_agent"]]:
    """
    Classify the user's query intent and route to the appropriate agent.

    This node uses structured LLM output to get a reliable IntentClassification
    dict, then uses Command-based routing to direct flow to the right agent.

    Args:
        state: Current RAG state containing the user query

    Returns:
        Command with classification update and routing decision
    """
    # Create structured LLM that returns IntentClassification
    structured_llm = _classifier_llm.with_structured_output(IntentClassification)

    # Format the prompt with the user's query
    prompt = CLASSIFICATION_PROMPT.format(query=state["query"])

    # Get classification
    classification: IntentClassification = structured_llm.invoke(prompt)

    # Route to the appropriate agent based on intent
    intent_to_agent = {
        "internal_docs": "internal_docs_agent",
        "elastic_docs": "elastic_docs_agent",
        "jira": "jira_agent",
    }

    next_agent = intent_to_agent[classification["intent"]]

    return Command(
        update={
            "intent": classification["intent"],
            "classification_reasoning": classification["reasoning"],
        },
        goto=next_agent,
    )
