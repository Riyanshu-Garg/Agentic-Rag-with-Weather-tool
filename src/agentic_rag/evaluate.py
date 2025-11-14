import os
from dotenv import load_dotenv
from langsmith.evaluation import evaluate
from langchain_core.messages import HumanMessage
import streamlit as st

from app import initialize_system
from config import AgentState, load_secrets_from_streamlit

# -------------------------------------------------
# INITIAL SETUP
# -------------------------------------------------
load_secrets_from_streamlit()

print("Initializing agent graph...")
app_graph, _, _, _, _ = initialize_system(uploaded_files=[])
print("Initialization complete.\n")

DATASET_NAME = "Neura_Dynamics_Assignment"   # <-- your dataset name in LangSmith


# -------------------------------------------------
# EXTRACT QUERY FROM LANGSMITH DATASET EXAMPLES
# -------------------------------------------------
def extract_query(example):
    """Extract 'current_query' from dataset items shaped like:
    { "input": { "current_query": "...", "messages": [...], ... } }
    """
    if isinstance(example, dict):
        inp = example.get("input")
        if isinstance(inp, dict):
            # Primary source
            if "current_query" in inp:
                return inp["current_query"]

            # Sometimes the first human message contains the query
            msgs = inp.get("messages")
            if isinstance(msgs, list) and len(msgs) > 0:
                first = msgs[0]
                if isinstance(first, dict) and "content" in first:
                    return first["content"]

    # fallback for unexpected shapes
    return None


# -------------------------------------------------
# SAFE ANSWER EXTRACTION FROM FINAL STATE
# -------------------------------------------------
def extract_answer(final_state):
    if isinstance(final_state, dict):
        for k in ("generated_answer", "answer", "output"):
            if k in final_state and final_state[k]:
                return final_state[k]

        msgs = final_state.get("messages")
        if isinstance(msgs, list) and msgs:
            last = msgs[-1]
            if isinstance(last, dict) and "content" in last:
                return last["content"]
            return str(last)

    return str(final_state)


# -------------------------------------------------
# MAIN PREDICTOR FOR LANGSMITH EVALUATION
# -------------------------------------------------
def run_agent_graph(example):
    print("\n=== NEW EXAMPLE RECEIVED ===")
    print("Example repr:", repr(example)[:1000])

    query = extract_query(example)
    if not query:
        return {"error": f"Could not extract query. Example: {repr(example)}"}

    # mock Streamlit session state
    class MockState:
        logs = []
    st.session_state = MockState()

    # Build agent state for your workflow
    initial_state = AgentState(
        messages=[HumanMessage(content=query)],
        chat_history=example.get("input", {}).get("chat_history", []),
        current_query=query,
        retrieved_docs=example.get("input", {}).get("retrieved_docs", [])
    )

    # Invoke your LangGraph workflow
    final_state = app_graph.invoke(initial_state)

    print("final_state repr:", repr(final_state)[:2000])
    answer = extract_answer(final_state)

    return {"output": answer}


# -------------------------------------------------
# RUN LANGSMITH EVALUATION WITH YOUR DATASET
# -------------------------------------------------
print(f"Running evaluation on LangSmith dataset: {DATASET_NAME}\n")

results = evaluate(
    run_agent_graph,
    data=DATASET_NAME,     # <-- THIS USES YOUR LANGSMITH DATASET
    description="Evaluation run for RAG + Weather Agent using LangGraph"
)

print("\n=== EVALUATION RESULTS ===")
print(results)
