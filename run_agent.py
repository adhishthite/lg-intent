"""
Run Script for the Email Agent
==============================

This script demonstrates how to:
1. Invoke a LangGraph agent
2. Stream execution to see each step as it completes
3. Handle interrupt() for human-in-the-loop workflows
4. Resume execution after human input

EXECUTION MODES:
----------------
LangGraph agents can be executed in two ways:

1. invoke() - Run to completion, return final state
   result = agent.invoke(initial_state, config)

2. stream() - Yield events as each node completes
   for event in agent.stream(initial_state, config, stream_mode="updates"):
       # Process each event

STREAM MODES:
-------------
- "values": Yields the full state after each node
- "updates": Yields only the state updates from each node (more efficient)
- "messages": Yields LLM messages as they're generated
- "custom": Yields custom events from nodes (using get_stream_writer())

CONFIGURATION:
--------------
The config dict must include a thread_id when using a checkpointer.
This thread_id identifies the conversation/session and allows:
- State persistence across calls
- Resuming from interrupts
- Multiple concurrent conversations

Example:
    config = {"configurable": {"thread_id": "unique-session-id"}}
"""

import uuid

from langgraph.types import Command

from email_agent import create_email_agent


def run_simple_email(agent, email_content: str, sender: str):
    """
    Run an email that completes without human intervention.

    This function demonstrates the basic execution flow:
    1. Create a config with a unique thread_id
    2. Build the initial state with all required fields
    3. Stream execution and print progress

    The email will flow through:
    read_email -> classify_intent -> [search_docs or bug_tracking] -> draft_response -> send_reply

    Because the urgency is "low", it skips human_review and goes directly to send_reply.

    Parameters:
    -----------
    agent: The compiled LangGraph agent
    email_content: The email body text to process
    sender: The sender's email address
    """
    # -------------------------------------------------------------------------
    # CONFIGURATION
    # -------------------------------------------------------------------------
    # The config dict tells LangGraph how to handle this execution.
    # The thread_id is REQUIRED when using a checkpointer - it identifies
    # this specific conversation/session.
    #
    # Each thread_id gets its own isolated state, so you can run multiple
    # conversations concurrently without them interfering.
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    # -------------------------------------------------------------------------
    # INITIAL STATE
    # -------------------------------------------------------------------------
    # This is the starting state for the agent. All fields defined in
    # EmailAgentState should be initialized here.
    #
    # Fields set to None will be populated by nodes during execution:
    # - classification: Set by classify_intent node
    # - search_results: Set by search_documentation node (if visited)
    # - customer_history: Set by bug_tracking node (if visited)
    # - response_text: Set by draft_response node
    initial_state = {
        # Input data - provided by the caller
        "email_content": email_content,
        "sender_email": sender,
        "email_id": str(uuid.uuid4())[:8],  # Short unique ID for this email
        # Intermediate data - will be set by nodes
        "classification": None,
        "search_results": None,
        "customer_history": None,
        # Output data - will be set by draft_response
        "response_text": None,
        "messages": None,
    }

    # Print header
    print(f"\n{'=' * 60}")
    print("PROCESSING EMAIL")
    print(f"{'=' * 60}")
    print(f"From: {sender}")
    print(f"Content: {email_content}")
    print(f"{'=' * 60}\n")

    # -------------------------------------------------------------------------
    # STREAM EXECUTION
    # -------------------------------------------------------------------------
    # Using stream() instead of invoke() lets us see progress in real-time.
    #
    # With stream_mode="updates", each event contains:
    # - The node name that just completed
    # - The state updates returned by that node
    #
    # Event structure: {"node_name": {"updated_key": new_value, ...}}
    for event in agent.stream(initial_state, config, stream_mode="updates"):
        for node_name, updates in event.items():
            print(f"[{node_name}] completed")
            # Show classification details when available
            if updates and "classification" in updates and updates["classification"]:
                c = updates["classification"]
                print(f"  -> Intent: {c['intent']}, Urgency: {c['urgency']}")


def run_email_with_review(agent, email_content: str, sender: str):
    """
    Run an email that requires human review (demonstrates interrupt).

    This function shows the HUMAN-IN-THE-LOOP pattern:

    PHASE 1: Initial Execution
    --------------------------
    1. Start the agent with initial state
    2. Agent runs through nodes until it hits human_review
    3. human_review calls interrupt() which PAUSES execution
    4. The stream ends (no more events)

    PHASE 2: Check for Interrupt
    ----------------------------
    1. Call agent.get_state(config) to get current state
    2. If state.next is not empty, we're paused at a node
    3. state.tasks contains info about pending nodes
    4. Each task may have interrupts with the interrupt payload

    PHASE 3: Resume Execution
    -------------------------
    1. Create a Command with resume={...} containing human's decision
    2. Call agent.stream(Command(...), config) to resume
    3. Agent continues from after the interrupt() call
    4. The human's input becomes the return value of interrupt()

    Parameters:
    -----------
    agent: The compiled LangGraph agent
    email_content: The email body text to process
    sender: The sender's email address
    """
    # Use a consistent thread_id so we can resume the same session
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # Build initial state (same as run_simple_email)
    initial_state = {
        "email_content": email_content,
        "sender_email": sender,
        "email_id": str(uuid.uuid4())[:8],
        "classification": None,
        "search_results": None,
        "customer_history": None,
        "response_text": None,
        "messages": None,
    }

    print(f"\n{'=' * 60}")
    print("PROCESSING EMAIL (with human review)")
    print(f"{'=' * 60}")
    print(f"From: {sender}")
    print(f"Content: {email_content}")
    print(f"{'=' * 60}\n")

    # -------------------------------------------------------------------------
    # PHASE 1: INITIAL EXECUTION (will pause at interrupt)
    # -------------------------------------------------------------------------
    # The agent will run until it hits interrupt() in human_review.
    # At that point, the stream will end without an error.
    for event in agent.stream(initial_state, config, stream_mode="updates"):
        for node_name, updates in event.items():
            print(f"[{node_name}] completed")
            if updates and "classification" in updates and updates["classification"]:
                c = updates["classification"]
                print(f"  -> Intent: {c['intent']}, Urgency: {c['urgency']}")

    # -------------------------------------------------------------------------
    # PHASE 2: CHECK FOR INTERRUPT
    # -------------------------------------------------------------------------
    # After the stream ends, we check if we're paused at an interrupt.
    #
    # get_state() returns a StateSnapshot with:
    # - values: The current state dict
    # - next: Tuple of node names that are pending (empty if done)
    # - tasks: List of pending task objects with interrupt info
    state = agent.get_state(config)

    # If state.next is not empty, we're paused waiting for input
    if state.next:
        print(f"\n{'=' * 60}")
        print("PAUSED FOR HUMAN REVIEW")
        print(f"{'=' * 60}")

        # -------------------------------------------------------------------------
        # ACCESS THE INTERRUPT PAYLOAD
        # -------------------------------------------------------------------------
        # The interrupt payload (what was passed to interrupt()) is available
        # in state.tasks[n].interrupts[m].value
        #
        # This is what the human sees to make their decision.
        for task in state.tasks:
            if hasattr(task, "interrupts") and task.interrupts:
                for interrupt_data in task.interrupts:
                    # interrupt_data.value is the dict passed to interrupt()
                    payload = interrupt_data.value
                    print(f"Draft response:\n{payload.get('draft_response', 'N/A')}")

        # -------------------------------------------------------------------------
        # SIMULATE HUMAN DECISION
        # -------------------------------------------------------------------------
        # In a real app, you would:
        # 1. Show a UI with the interrupt payload
        # 2. Let the human edit the response or approve/reject
        # 3. Collect their input into a dict
        #
        # Here we just auto-approve for the demo
        print("\n[Human] Approving the response...")

        # -------------------------------------------------------------------------
        # PHASE 3: RESUME EXECUTION
        # -------------------------------------------------------------------------
        # To resume, we call stream() with a Command that has resume={...}
        #
        # Command(resume={"approved": True})
        #
        # The dict passed to resume becomes the return value of interrupt()
        # in the human_review node. The node then processes this input and
        # routes to the next node (send_reply or END).
        human_input = Command(resume={"approved": True})

        # Stream the resumed execution
        # We use the SAME config to continue the same thread
        for event in agent.stream(human_input, config, stream_mode="updates"):
            for node_name, updates in event.items():
                print(f"[{node_name}] completed")


def main():
    """
    Main entry point - runs three example emails.

    Example 1: Simple Question
    --------------------------
    - Low urgency password reset question
    - Routes: classify_intent -> search_documentation -> draft_response -> send_reply
    - No human review needed (low urgency)

    Example 2: Billing Complaint
    ----------------------------
    - High urgency billing issue
    - Routes: classify_intent -> draft_response -> human_review -> send_reply
    - Requires human review (billing + high urgency)
    - Demonstrates interrupt() and resume with Command

    Example 3: Bug Report
    ---------------------
    - High urgency crash report
    - Routes: classify_intent -> bug_tracking -> draft_response -> human_review -> send_reply
    - Requires human review (high urgency)
    - Shows bug_tracking node creating a bug ID
    """
    print("Creating email agent...")
    agent = create_email_agent()

    # -------------------------------------------------------------------------
    # EXAMPLE 1: SIMPLE QUESTION
    # -------------------------------------------------------------------------
    # This email has low urgency, so it will complete without human review.
    # It will go through search_documentation to gather relevant docs.
    run_simple_email(
        agent,
        email_content="Hi, I was wondering how to reset my password? Thanks!",
        sender="user@example.com",
    )

    # -------------------------------------------------------------------------
    # EXAMPLE 2: BILLING COMPLAINT
    # -------------------------------------------------------------------------
    # Billing issues are routed directly to draft_response (no doc search needed).
    # The high urgency triggers human_review, which demonstrates interrupt().
    run_email_with_review(
        agent,
        email_content="I've been charged twice for my subscription this month. Please fix this immediately!",
        sender="premium.user@example.com",
    )

    # -------------------------------------------------------------------------
    # EXAMPLE 3: BUG REPORT
    # -------------------------------------------------------------------------
    # Bug reports go through bug_tracking to create a ticket.
    # The crash severity makes it high urgency, triggering human_review.
    # The response includes the generated bug ID from bug_tracking.
    run_email_with_review(
        agent,
        email_content="The export feature crashes when I try to export more than 100 items.",
        sender="developer@example.com",
    )


if __name__ == "__main__":
    main()
