import streamlit as st
import time
import os
import functools
from config import SECRETS, AgentState
from vectorstore import load_uploaded_docs, split_documents, build_qdrant_vectorstore, calculate_knowledge_hash
from agents import router_agent, retrieve_agent, weather_search_agent, generate_agent, route_decision
import warnings
from langgraph.graph import START, END, StateGraph
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langchain_core.messages import HumanMessage, AIMessage


warnings.filterwarnings("ignore")
def initialize_system(uploaded_files, chunk_size=250, k=3, temperature=0.0):
    docs = load_uploaded_docs(uploaded_files)

    if not docs:
        try:
            base_dir = os.path.dirname(__file__)
            pdf_path = os.path.join(base_dir, "pdf_file", "Riyanshu_Resume.pdf")
            if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 0:
                from langchain_community.document_loaders import PyPDFLoader
                loader = PyPDFLoader(pdf_path)
                file_docs = loader.load()
                for doc in file_docs:
                    if not hasattr(doc, "metadata") or doc.metadata is None:
                        doc.metadata = {}
                    doc.metadata["source"] = os.path.basename(pdf_path)
                docs.extend(file_docs)
            else:
                st.warning("Default dOCUMENT not found or is empty. Continuing without docs.")
        except Exception as e:
            st.error(f"Failed to load default resume: {e}")

    doc_splits = split_documents(docs, chunk_size=chunk_size)

    retriever, retriever_tool = build_qdrant_vectorstore(
        doc_splits,
        google_api_key=SECRETS["GOOGLE_API_KEY"],
        qdrant_url=SECRETS["QDRANT_URL"],
        qdrant_api=SECRETS["QDRANT_API"]
    )

    weather_search_tool = OpenWeatherMapAPIWrapper()

    # --- BIND AGENTS TO TOOLS USING functools.partial ---
    router_node = functools.partial(router_agent, temperature=temperature)
    retrieve_node = functools.partial(retrieve_agent, retriever_instance=retriever)
    weather_search_node = functools.partial(
        weather_search_agent, 
        weather_search_tool=weather_search_tool, 
        temperature=temperature
    )
    generate_node = functools.partial(generate_agent, temperature=temperature)
    # --- END BINDING ---

    workflow = StateGraph(AgentState)
    
    # --- USE THE BOUND NODES ---
    workflow.add_node("router", router_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("weather_search", weather_search_node)
    workflow.add_node("generate", generate_node)
    # --- END ---

    workflow.add_edge(START, "router")

    workflow.add_conditional_edges(
        "router",
        route_decision,
        {"retrieve": "retrieve", "weather_search": "weather_search"}
    )

    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("weather_search", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile(), retriever, weather_search_tool, temperature, retriever_tool


def run_app():
    st.set_page_config(page_title="RAG WITH WEATHER AGENT INTEGRATION", layout="wide")
    st.title("RAG WITH WEATHER AGENT INTEGRATION")

    # Sidebar UI
    with st.sidebar:
        st.markdown(f"""
        **Current Settings:**  
        - Chunk Size: `{st.session_state.get('chunk_size', 300)}`
        - Retriever K: `{st.session_state.get('retriever_k', 5)}`
        - Temperature: `{st.session_state.get('temperature', 0.0)}`
        """)
        with st.expander("⚙️ Adjust Configuration", expanded=False):
            chunk_size = st.slider("Text Chunk Size", 100, 2000, st.session_state.get('chunk_size', 300), 50)
            retriever_k = st.slider("Retriever K Value (Top K Docs)", 1, 10, st.session_state.get('retriever_k', 5))
            temperature = st.slider("LLM Temperature", 0.0, 1.0, st.session_state.get('temperature', 0.0), 0.1)
            st.divider()

        st.subheader("Knowledge Sources")
        uploaded_files = st.file_uploader("Upload text files", type=["txt", "pdf", "docx"], accept_multiple_files=True)
        reset_params = st.button("Apply Parameters & Update Knowledge")
        st.divider()

        st.subheader("Agent Roles")
        st.markdown("""
        - **Router**: Determines workflow path based on query type.
        - **Retriever**: Fetches relevant documents from internal knowledge base.
        - **weather Searcher**: Finds real-time weather information.
        - **Generator**: Creates the final answer.
        """)

    # Initialize session state
    if "logs" not in st.session_state:
        st.session_state.logs = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "final_answer" not in st.session_state:
        st.session_state.final_answer = ""
    if "params_applied" not in st.session_state:
        st.session_state.params_applied = False
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.0
    if "knowledge_hash" not in st.session_state:
        st.session_state.knowledge_hash = ""

    # Calculate current knowledge hash
    current_knowledge_hash = calculate_knowledge_hash(uploaded_files or [])
    knowledge_changed = current_knowledge_hash != st.session_state.knowledge_hash

    # Initialize or update system
    if (reset_params or not st.session_state.params_applied or knowledge_changed):
        with st.spinner("Configuring system with new parameters and knowledge sources..."):
            try:
                st.session_state.graph, st.session_state.retriever_instance, st.session_state.weather_search_tool, st.session_state.temperature, st.session_state.retriever_tool_for_display = initialize_system(
                    uploaded_files=uploaded_files or [],
                    chunk_size=chunk_size,
                    k=retriever_k,
                    temperature=temperature
                )
                st.session_state.params_applied = True
                st.session_state.knowledge_hash = current_knowledge_hash
                st.success("System configured with new knowledge!")
            except Exception as e:
                st.error(f"Configuration failed: {str(e)}")
                st.stop()

    # Chat interface
    with st.container():
        st.subheader("Chat")
        for msg in st.session_state.chat_history:
            if isinstance(msg, HumanMessage):
                with st.chat_message("user"):
                    st.write(msg.content)
            elif isinstance(msg, AIMessage):
                with st.chat_message("assistant"):
                    st.write(msg.content)

        if prompt := st.chat_input("Ask about your knowledge sources..."):
            user_msg = HumanMessage(content=prompt)
            st.session_state.chat_history.append(user_msg)

            with st.chat_message("user"):
                st.write(prompt)

            agent_state = AgentState(
                messages=[user_msg],
                chat_history=st.session_state.chat_history,
                current_query=prompt,
                retrieved_docs=[]
            )

            with st.spinner("Executing workflow..."):
                st.session_state.logs = [f"New query: {prompt}"]
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    step_count = 0
                    max_steps = 10
                    current_state_updates = {}

                    for output in st.session_state.graph.stream(agent_state):
                        node_name = list(output.keys())[0]
                        node_state = output[node_name]

                        status_text.info(f"Executing: **{node_name.replace('_', ' ').title()}**")
                        st.session_state.logs.append(f"Completed node: {node_name}")

                        step_count += 1
                        progress_bar.progress(min(step_count / max_steps, 1.0))
                        time.sleep(0.3)

                        current_state_updates = node_state


                    progress_bar.empty()
                    status_text.success("✅ Workflow completed!")

                    if current_state_updates and 'generated_answer' in current_state_updates:
                        final_answer = current_state_updates['generated_answer']
                        ai_msg = AIMessage(content=final_answer)
                        st.session_state.chat_history.append(ai_msg)
                        st.session_state.final_answer = final_answer

                        with st.chat_message("assistant"):
                            st.write(final_answer)
                    else:
                        fallback_message = "I couldn't generate a complete response for your query. Please try rephrasing."
                        ai_msg = AIMessage(content=fallback_message)
                        st.session_state.chat_history.append(ai_msg)
                        st.session_state.final_answer = fallback_message

                        with st.chat_message("assistant"):
                            st.write(fallback_message)

                except Exception as e:
                    import traceback
                    error_trace = traceback.format_exc()
                    status_text.error(f"❌ Execution failed: {str(e)}")
                    st.session_state.logs.append(f"ERROR TRACEBACK:{error_trace}")
                    error_msg = AIMessage(content="Sorry, I encountered an error processing your request. Please check the logs.")
                    st.session_state.chat_history.append(error_msg)

                    with st.chat_message("assistant"):
                        st.write("Sorry, I encountered an error processing your request. Please check the logs.")

    # Display results
    with st.expander("Execution Details", expanded=False):
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Workflow Log")
            log_container = st.container()
            for log in st.session_state.logs:
                log_container.code(log, language="log")

            st.subheader("Active Knowledge Sources")
            if uploaded_files:
                st.markdown("**Uploaded Files:**")
                for file in uploaded_files:
                    st.markdown(f"- {file.name}")
            else:
                st.markdown("Using default knowledge sources")

        with col2:
            st.subheader("Workflow Diagram")
            st.graphviz_chart("""
                digraph {
                    node [shape=box, style=rounded]
                    start -> router
                    router -> retrieve [label="retrieve"]
                    router -> weather_search [label="weather_search"]
                    retrieve -> generate [label="docs retrieved"]
                    weather_search -> generate [label="results found"]
                    generate -> end [label="answer created"]
                }
            """)

            st.subheader("Agent Configuration")
            st.markdown(f"""
            - **Chunk Size**: `{chunk_size}`
            - **Retriever K (Top K Docs)**: `{retriever_k}`
            - **LLM Temperature**: `{temperature}`
            - **Main LLM Model**: `llama3-70b-8192` (Groq)
            - **Grading LLM Model**: `gemma2-9b-it` (Groq)
            """)

    # Add reset button
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.logs = []
        st.session_state.final_answer = ""
        st.session_state.get("knowledge_hash", None)
        st.session_state.params_applied = False
        st.rerun()


if __name__ == '__main__':
    run_app()