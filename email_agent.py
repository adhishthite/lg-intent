"""
Email Agent using LangGraph - Thinking in LangGraph Tutorial
=============================================================

This implements a customer support email agent following the "Thinking in LangGraph"
methodology from the official LangGraph documentation.

CORE CONCEPTS DEMONSTRATED:
---------------------------
1. STATE: A shared TypedDict that flows between all nodes. Each node reads from
   state and returns updates to it. State persists across the entire workflow.

2. NODES: Functions that do work. Each node:
   - Receives the current state as input
   - Performs some operation (LLM call, API call, data processing)
   - Returns either a dict (state updates) or a Command (updates + routing)

3. COMMAND: An object that combines state updates WITH routing decisions.
   Instead of defining edges externally, nodes can decide where to go next.
   Format: Command(update={"key": value}, goto="next_node_name")

4. INTERRUPT: Pauses execution for human input. The workflow saves its state,
   returns control to the caller, and resumes exactly where it left off when
   the human provides input via Command(resume={...}).

5. EDGES: Connections between nodes. With Command-based routing, you only need
   to define essential edges (start, end). Nodes handle conditional routing.

WORKFLOW VISUALIZATION:
-----------------------
                    __start__
                        |
                   read_email
                        |
                classify_intent -----> Uses LLM to classify, then routes via Command
                /       |       \
     bug_tracking  search_docs   |
                \       |       /
                 draft_response -----> Generates reply, routes based on urgency
                   /         \
           human_review       |  -----> interrupt() pauses here for human input
                   \         /
                   send_reply
                        |
                    __end__
"""

from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, RetryPolicy, interrupt
from typing_extensions import TypedDict

# Load environment variables from .env file (expects OPENAI_API_KEY)
load_dotenv()


# =============================================================================
# STEP 1: DEFINE THE STATE SCHEMA
# =============================================================================
#
# State is the "memory" that flows between nodes. Key principles:
#
# 1. STORE RAW DATA, NOT FORMATTED TEXT
#    - Don't store pre-formatted prompts in state
#    - Each node formats data differently for its own needs
#    - This keeps state clean and flexible
#
# 2. ONLY STORE WHAT PERSISTS ACROSS STEPS
#    - Don't store temporary/intermediate values
#    - Don't store values that can be derived from other state
#
# 3. USE TYPED STRUCTURES
#    - TypedDict gives you type checking and IDE support
#    - Literal types constrain values to valid options
# =============================================================================


class EmailClassification(TypedDict):
    """
    Structure for email classification results.

    This TypedDict is used with `llm.with_structured_output()` to get
    the LLM to return a properly structured dict instead of free text.

    The Literal types ensure the LLM only returns valid values.
    """

    # The type of email - determines which processing path to take
    intent: Literal["question", "bug", "billing", "feature", "complex"]

    # How urgent is this email - affects whether human review is needed
    urgency: Literal["low", "medium", "high", "critical"]

    # Brief description of what the email is about
    topic: str

    # One-sentence summary for quick reference
    summary: str


class EmailAgentState(TypedDict):
    """
    The main state schema for our email agent.

    This TypedDict defines ALL data that flows between nodes.
    Each field represents a piece of information that:
    - Is set by one or more nodes
    - May be read by downstream nodes
    - Persists for the lifetime of a single email processing run

    DESIGN PRINCIPLE: Store raw data, not formatted text.
    Different nodes can format the same data differently for their needs.
    For example, search_results is a list of strings - the draft_response
    node formats it into a bullet list when building its prompt.
    """

    # -------------------------------------------------------------------------
    # INPUT DATA (set at the start, read by multiple nodes)
    # -------------------------------------------------------------------------

    # The raw email body text - the primary input to process
    email_content: str

    # Who sent the email - used for personalization and reply addressing
    sender_email: str

    # Unique identifier for this email - used for tracking/logging
    email_id: str

    # -------------------------------------------------------------------------
    # INTERMEDIATE DATA (set by processing nodes, read by downstream nodes)
    # -------------------------------------------------------------------------

    # Classification result from the classify_intent node
    # None initially, set after classification completes
    classification: EmailClassification | None

    # Search results from documentation lookup
    # None if we didn't go through the search_documentation node
    search_results: list[str] | None

    # Customer/bug data from external systems
    # None if we didn't go through the bug_tracking node
    customer_history: dict | None

    # -------------------------------------------------------------------------
    # OUTPUT DATA (set by draft_response, possibly modified by human_review)
    # -------------------------------------------------------------------------

    # The generated email response text
    response_text: str | None

    # Message history (for potential multi-turn conversations)
    messages: list[str] | None


# =============================================================================
# STEP 2: INITIALIZE THE LLM
# =============================================================================
#
# We use a single LLM instance that's shared across nodes.
# - model: The OpenAI model to use
# - temperature=0: Deterministic outputs for consistent classification
# =============================================================================

llm = ChatOpenAI(model="gpt-5-nano", temperature=0)


# =============================================================================
# STEP 3: IMPLEMENT NODE FUNCTIONS
# =============================================================================
#
# Each node is a function that:
# 1. Takes the current state as its only argument
# 2. Does some work (LLM calls, API calls, data processing)
# 3. Returns updates to apply to state
#
# RETURN TYPES:
# - dict: Simple state update. LangGraph applies the update and follows
#         any edges defined in the graph to determine the next node.
#
# - Command: State update + routing decision. Allows the node itself to
#            decide where to go next based on its results.
#            Format: Command(update={...}, goto="next_node_name")
#
# TYPE HINTS FOR COMMAND:
# - Command[Literal["node_a", "node_b"]] tells LangGraph which nodes this
#   node can route to. This is used to infer edges for visualization.
# =============================================================================


def read_email(state: EmailAgentState) -> dict:
    """
    NODE: read_email
    ================
    PURPOSE: Extract and parse email content from the source.

    IN A REAL SYSTEM, this node would:
    - Connect to Gmail API, Outlook, or email service
    - Parse email headers (From, Subject, Date, etc.)
    - Extract the email body (handle HTML vs plain text)
    - Download and process attachments
    - Extract metadata (thread ID, labels, etc.)

    INPUTS (from state):
    - email_content: The raw email text (in this demo, already provided)

    OUTPUTS (state updates):
    - messages: Adds a HumanMessage to track what we're processing

    ROUTING:
    - Returns a simple dict, so routing follows the explicit edge
    - Next node: classify_intent (defined via add_edge in graph setup)
    """
    return {
        "messages": [
            HumanMessage(content=f"Processing email: {state['email_content']}")
        ]
    }


def classify_intent(
    state: EmailAgentState,
) -> Command[Literal["search_documentation", "draft_response", "bug_tracking"]]:
    """
    NODE: classify_intent
    =====================
    PURPOSE: Use an LLM to analyze the email and determine its type and urgency.

    This is a KEY NODE that demonstrates two important LangGraph patterns:

    PATTERN 1: STRUCTURED OUTPUT
    ----------------------------
    Instead of asking the LLM for free text and parsing it, we use
    `llm.with_structured_output(EmailClassification)` which:
    - Tells the LLM to return a JSON object matching our TypedDict
    - Automatically parses the response into a Python dict
    - Validates that required fields are present
    - Ensures Literal values are one of the allowed options

    PATTERN 2: COMMAND-BASED ROUTING
    --------------------------------
    Instead of defining conditional edges externally, this node decides
    where to go next based on its classification results:
    - question/feature -> search_documentation (need to look up docs)
    - bug -> bug_tracking (need to create/lookup bug ticket)
    - billing/complex -> draft_response (go directly to drafting)

    The Command object bundles the state update AND the routing decision:
    Command(update={"classification": ...}, goto="next_node")

    INPUTS (from state):
    - email_content: The email text to classify
    - sender_email: Who sent it (might affect classification)

    OUTPUTS (state updates):
    - classification: The EmailClassification dict with intent, urgency, etc.

    ROUTING (via Command):
    - "search_documentation" for questions and feature requests
    - "bug_tracking" for bug reports
    - "draft_response" for billing, complex, or other types
    """
    # Create a structured LLM that returns EmailClassification dict
    # This uses OpenAI's function calling under the hood
    structured_llm = llm.with_structured_output(EmailClassification)

    # FORMAT THE PROMPT ON-DEMAND
    # ---------------------------
    # Notice we don't store this prompt in state - we build it here
    # from the raw data. This is the "format prompts inside nodes" principle.
    classification_prompt = f"""
    Analyze this customer email and classify it:

    Email: {state["email_content"]}
    From: {state["sender_email"]}

    Provide classification including:
    - intent: one of [question, bug, billing, feature, complex]
    - urgency: one of [low, medium, high, critical]
    - topic: brief topic description
    - summary: one sentence summary
    """

    # Invoke the LLM - returns a dict matching EmailClassification
    classification = structured_llm.invoke(classification_prompt)

    # ROUTING LOGIC
    # -------------
    # Based on the classification, decide which node to go to next.
    # All paths eventually lead to draft_response, but they may
    # gather additional context first.
    if classification["intent"] in ["question", "feature"]:
        # Questions and feature requests need documentation lookup
        goto = "search_documentation"
    elif classification["intent"] == "bug":
        # Bug reports need to be logged in the bug tracking system
        goto = "bug_tracking"
    else:
        # Billing, complex, or other - go directly to drafting
        # The draft_response node will decide if human review is needed
        goto = "draft_response"

    # Return Command with both the state update AND routing decision
    return Command(update={"classification": classification}, goto=goto)


def search_documentation(
    state: EmailAgentState,
) -> Command[Literal["draft_response"]]:
    """
    NODE: search_documentation
    ==========================
    PURPOSE: Search internal documentation for relevant information to include
    in the response.

    IN A REAL SYSTEM, this node would:
    - Connect to a vector database (Pinecone, Weaviate, Chroma, etc.)
    - Embed the query using the same model as the documents
    - Perform similarity search to find relevant docs
    - Optionally re-rank results for relevance
    - Return the top N most relevant chunks

    This is a "DATA STEP" - it retrieves external information but doesn't
    use an LLM for reasoning. It's purely about data retrieval.

    INPUTS (from state):
    - classification: Used to understand what topic to search for

    OUTPUTS (state updates):
    - search_results: List of relevant documentation snippets

    ROUTING (via Command):
    - Always goes to "draft_response" (only one possible destination)

    NOTE: This node has a RetryPolicy in the graph setup, meaning if it
    fails (e.g., vector DB timeout), LangGraph will automatically retry
    up to 3 times before giving up.
    """
    classification = state["classification"]

    # MOCK IMPLEMENTATION
    # In production, replace this with actual vector search
    mock_results = [
        f"Documentation about {classification['topic']}",
        "FAQ: Common questions and answers",
        "Troubleshooting guide for this feature",
    ]

    # Always route to draft_response after gathering docs
    return Command(update={"search_results": mock_results}, goto="draft_response")


def bug_tracking(state: EmailAgentState) -> Command[Literal["draft_response"]]:
    """
    NODE: bug_tracking
    ==================
    PURPOSE: Create or lookup a bug in the tracking system, and gather
    customer history for context.

    IN A REAL SYSTEM, this node would:
    - Check if a similar bug already exists (fuzzy matching)
    - Create a new bug ticket if needed (Jira, GitHub Issues, Linear, etc.)
    - Lookup customer's previous bug reports and support history
    - Check customer tier/SLA for prioritization
    - Assign the bug to the appropriate team

    This is both a "DATA STEP" (retrieving customer info) and an
    "ACTION STEP" (creating a bug ticket in an external system).

    INPUTS (from state):
    - email_id: Used to generate a unique bug ID
    - classification: Used to set bug priority and description

    OUTPUTS (state updates):
    - customer_history: Dict with customer info and bug ticket details

    ROUTING (via Command):
    - Always goes to "draft_response"
    """
    classification = state["classification"]

    # MOCK IMPLEMENTATION
    # In production, replace with actual Jira/GitHub API calls
    customer_history = {
        "previous_bugs": 2,  # Customer has reported 2 bugs before
        "tier": "premium",  # Customer tier affects response priority
        "bug_id": f"BUG-{state['email_id'][:8]}",  # Generated bug ticket ID
        "bug_topic": classification["topic"],  # What the bug is about
    }

    return Command(update={"customer_history": customer_history}, goto="draft_response")


def draft_response(
    state: EmailAgentState,
) -> Command[Literal["human_review", "send_reply"]]:
    """
    NODE: draft_response
    ====================
    PURPOSE: Generate an email response using all gathered context, then
    decide whether human review is needed.

    This is the CENTRAL NODE where all processing paths converge.
    It receives context from whichever upstream path was taken:
    - search_results (if we went through search_documentation)
    - customer_history (if we went through bug_tracking)
    - Just classification (if we came directly)

    KEY PATTERN: FORMAT PROMPTS ON-DEMAND
    -------------------------------------
    Notice how we build the prompt dynamically from raw state data:
    - We check if search_results exists and format it as a bullet list
    - We check if customer_history exists and extract relevant fields
    - We combine everything into a single prompt

    This is better than storing pre-formatted text because:
    - Different nodes might need the same data formatted differently
    - We can change prompt format without changing state schema
    - State stays clean and contains only raw data

    ROUTING DECISION:
    -----------------
    This node decides whether to go to human_review or send_reply based on:
    - Urgency: high/critical emails need human review
    - Intent: "complex" emails need human review
    - (In production, you might add more criteria)

    INPUTS (from state):
    - email_content: Original email to respond to
    - classification: Intent and urgency for context
    - search_results: Documentation to reference (optional)
    - customer_history: Customer context (optional)

    OUTPUTS (state updates):
    - response_text: The generated email response

    ROUTING (via Command):
    - "human_review" if high urgency or complex
    - "send_reply" if safe to send automatically
    """
    classification = state["classification"]

    # BUILD CONTEXT SECTIONS FROM RAW STATE DATA
    # ------------------------------------------
    # We format the raw data into prompt-friendly text here, not in state
    context_sections = []

    # Add documentation context if available
    if state.get("search_results"):
        formatted_docs = "\n".join(f"- {doc}" for doc in state["search_results"])
        context_sections.append(f"Relevant documentation:\n{formatted_docs}")

    # Add customer context if available
    if state.get("customer_history"):
        history = state["customer_history"]
        context_sections.append(f"Customer tier: {history.get('tier', 'standard')}")
        if "bug_id" in history:
            context_sections.append(f"Bug ID: {history['bug_id']}")

    # BUILD THE PROMPT
    # ----------------
    # Combine all context into a single prompt for the LLM
    draft_prompt = f"""
    Draft a response to this customer email:
    {state["email_content"]}

    Email intent: {classification["intent"]}
    Urgency level: {classification["urgency"]}

    {chr(10).join(context_sections)}

    Guidelines:
    - Be professional and helpful
    - Address their specific concern
    - Use the provided documentation when relevant
    - Keep the response concise but thorough
    """

    # Generate the response
    response = llm.invoke([HumanMessage(content=draft_prompt)])

    # ROUTING DECISION
    # ----------------
    # Determine if human review is needed based on urgency and complexity
    needs_review = (
        classification["urgency"] in ["high", "critical"]
        or classification["intent"] == "complex"
    )

    next_node = "human_review" if needs_review else "send_reply"

    return Command(update={"response_text": response.content}, goto=next_node)


def human_review(state: EmailAgentState) -> Command[Literal["send_reply"]]:
    """
    NODE: human_review
    ==================
    PURPOSE: Pause execution for a human to review and approve/edit the response.

    This node demonstrates HUMAN-IN-THE-LOOP using LangGraph's interrupt() function.

    HOW INTERRUPT WORKS:
    --------------------
    1. When interrupt() is called, execution STOPS immediately
    2. The graph's state is SAVED (via the checkpointer)
    3. Control returns to the caller with the interrupt payload
    4. The caller (your app) shows the payload to a human
    5. When the human decides, you call graph.stream(Command(resume={...}))
    6. Execution RESUMES from right after the interrupt() call
    7. The human's input is returned as the value of interrupt()

    CRITICAL RULE: interrupt() MUST COME FIRST
    ------------------------------------------
    Any code BEFORE interrupt() will RE-RUN when execution resumes.
    This is because LangGraph replays from the start of the node.
    So always put interrupt() at the very beginning of the node,
    or at least before any side effects you don't want repeated.

    THE INTERRUPT PAYLOAD:
    ----------------------
    The dict passed to interrupt() is what the human will see.
    Include everything they need to make a decision:
    - The original email
    - The draft response
    - Classification info
    - Any relevant context

    RESUMING EXECUTION:
    -------------------
    When you call graph.stream(Command(resume={...}), config), the dict
    you pass to resume becomes the return value of interrupt().
    In this case, we expect:
    - approved: bool - whether to send the response
    - edited_response: str (optional) - modified response text

    INPUTS (from state):
    - email_id, email_content, response_text, classification

    OUTPUTS (state updates):
    - response_text: Possibly updated with human's edits

    ROUTING (via Command):
    - "send_reply" if approved
    - END if rejected (human will handle directly)
    """
    classification = state["classification"]

    # INTERRUPT FOR HUMAN INPUT
    # -------------------------
    # This pauses execution and returns the payload to the caller.
    # The human's response becomes the return value.
    human_decision = interrupt(
        {
            # Include everything the human needs to make a decision
            "email_id": state["email_id"],
            "original_email": state["email_content"],
            "draft_response": state["response_text"],
            "urgency": classification["urgency"],
            "intent": classification["intent"],
            "action": "Please review and approve/edit this response",
        }
    )

    # CODE AFTER INTERRUPT RUNS AFTER RESUME
    # --------------------------------------
    # At this point, human_decision contains what the human provided

    if human_decision.get("approved", True):
        # Human approved - use their edited version if provided
        edited_response = human_decision.get("edited_response", state["response_text"])
        return Command(update={"response_text": edited_response}, goto="send_reply")
    else:
        # Human rejected - they'll handle this email directly
        # Route to END to terminate the workflow
        return Command(update={}, goto=END)


def send_reply(state: EmailAgentState) -> dict:
    """
    NODE: send_reply
    ================
    PURPOSE: Send the final email response to the customer.

    This is an "ACTION STEP" - it performs an external action (sending email)
    but doesn't modify state or make routing decisions.

    IN A REAL SYSTEM, this node would:
    - Connect to email service (Gmail API, SendGrid, SES, etc.)
    - Format the email with proper headers
    - Add signature, disclaimers, unsubscribe links
    - Handle attachments if needed
    - Log the sent email for compliance
    - Update CRM with interaction record

    RETURN TYPE: dict (not Command)
    --------------------------------
    This node returns a simple dict (empty in this case) because:
    - It doesn't need to update state
    - It doesn't need to make routing decisions
    - The next step is END, defined via add_edge in graph setup

    INPUTS (from state):
    - sender_email: Who to send the reply to
    - email_id: For the subject line
    - response_text: The email body to send

    OUTPUTS (state updates):
    - None (returns empty dict)

    ROUTING:
    - Follows the explicit edge to END
    """
    # MOCK IMPLEMENTATION
    # In production, replace with actual email sending
    print(f"\n{'=' * 60}")
    print("SENDING EMAIL REPLY")
    print(f"{'=' * 60}")
    print(f"To: {state['sender_email']}")
    print(f"Subject: Re: Support Request {state['email_id']}")
    print(f"\n{state['response_text']}")
    print(f"{'=' * 60}\n")

    # Return empty dict - no state updates needed
    return {}


# =============================================================================
# STEP 4: WIRE THE GRAPH TOGETHER
# =============================================================================
#
# Now we connect our nodes into a working graph using StateGraph.
#
# KEY CONCEPTS:
#
# 1. StateGraph(EmailAgentState)
#    - Creates a graph builder with our state schema
#    - The schema defines what data flows between nodes
#
# 2. add_node("name", function)
#    - Registers a node with a name and its implementation
#    - The name is used for routing (in Command(goto="name"))
#
# 3. add_edge(from_node, to_node)
#    - Creates a fixed edge between two nodes
#    - Used for unconditional transitions
#
# 4. START and END
#    - Special nodes representing graph entry and exit
#    - START -> first_node defines where execution begins
#    - last_node -> END defines where execution can terminate
#
# 5. RetryPolicy
#    - Automatically retry a node if it fails
#    - Useful for nodes that call external APIs
#
# 6. Checkpointer (MemorySaver)
#    - Saves graph state between steps
#    - REQUIRED for interrupt() to work
#    - In production, use a persistent store (Redis, PostgreSQL, etc.)
#
# WHY SO FEW EDGES?
# -----------------
# We only define 3 edges:
# - START -> read_email (entry point)
# - read_email -> classify_intent (fixed sequence)
# - send_reply -> END (exit point)
#
# All other routing is handled by Command objects inside nodes!
# This is the "Command-based routing" pattern - nodes decide where
# to go based on their results, rather than having external edge logic.
# =============================================================================


def create_email_agent():
    """
    Create and compile the email agent graph.

    This function:
    1. Creates a StateGraph with our state schema
    2. Adds all nodes with their implementations
    3. Defines the essential edges
    4. Compiles with a checkpointer for persistence

    Returns a compiled graph that can be invoked with .invoke() or .stream()
    """
    # Create the graph builder with our state schema
    workflow = StateGraph(EmailAgentState)

    # -------------------------------------------------------------------------
    # ADD NODES
    # -------------------------------------------------------------------------
    # Each add_node call registers a node with a name and function

    # Entry point - reads and parses the email
    workflow.add_node("read_email", read_email)

    # Classification - analyzes email and routes to appropriate handler
    workflow.add_node("classify_intent", classify_intent)

    # Documentation search - has retry policy for transient failures
    # RetryPolicy automatically retries up to 3 times if the node fails
    workflow.add_node(
        "search_documentation",
        search_documentation,
        retry=RetryPolicy(max_attempts=3),
    )

    # Bug tracking - creates/looks up bug tickets
    workflow.add_node("bug_tracking", bug_tracking)

    # Response drafting - generates the email response
    workflow.add_node("draft_response", draft_response)

    # Human review - pauses for human approval (uses interrupt)
    workflow.add_node("human_review", human_review)

    # Send reply - final action node
    workflow.add_node("send_reply", send_reply)

    # -------------------------------------------------------------------------
    # ADD EDGES
    # -------------------------------------------------------------------------
    # We only need essential edges - nodes handle conditional routing via Command

    # Entry: START -> read_email
    workflow.add_edge(START, "read_email")

    # Fixed sequence: read_email -> classify_intent
    workflow.add_edge("read_email", "classify_intent")

    # Exit: send_reply -> END
    workflow.add_edge("send_reply", END)

    # NOTE: All other routing is handled by Command objects in nodes!
    # - classify_intent routes to search_documentation, bug_tracking, or draft_response
    # - search_documentation routes to draft_response
    # - bug_tracking routes to draft_response
    # - draft_response routes to human_review or send_reply
    # - human_review routes to send_reply or END

    # -------------------------------------------------------------------------
    # COMPILE WITH CHECKPOINTER
    # -------------------------------------------------------------------------
    # The checkpointer saves state between steps, which is REQUIRED for:
    # - interrupt() to work (state must be saved when we pause)
    # - Resuming from failures
    # - Long-running workflows

    # MemorySaver stores state in memory (lost on restart)
    # In production, use SqliteSaver, PostgresSaver, or RedisSaver
    memory = MemorySaver()

    # Compile the graph - this validates the graph and returns a runnable
    return workflow.compile(checkpointer=memory)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # See run_agent.py for examples of how to invoke the agent
    print("Use 'make run' to execute the email agent demo")
    print("Or import create_email_agent() and invoke it programmatically")
