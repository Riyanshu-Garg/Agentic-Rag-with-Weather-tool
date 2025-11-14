import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_classic import hub
from langchain_core.output_parsers import StrOutputParser
from typing import Dict, Any
from config import SECRETS, AgentState


def router_agent(state: AgentState, temperature: float) -> dict:
    st.session_state.logs.append("---ROUTER AGENT---")
    model = ChatGroq(
        temperature=temperature,
        model_name="openai/gpt-oss-20b",
        groq_api_key=SECRETS["GROQ_API_KEY"]
    )

    prompt = PromptTemplate(
        template="""As the Router Agent, analyze the user's question and conversation history to determine the best next step.

        Conversation History:
        {history}

        Current Question: {question}

        Choose one of these actions:
        - "retrieve": If question can be answered with known documents (from internal knowledge base)
        - "weather_search": If question information about weather or climate for any location

        Only respond with the action word.""",
        input_variables=["question", "history"]
    )

    history_str = "".join([f"{m.type}: {m.content}" for m in state.chat_history[-5:]])
    response = model.invoke(prompt.format(question=state.current_query, history=history_str))
    decision = response.content.strip().lower()

    st.session_state.logs.append(f"Routing decision: {decision}")
    return {"next_step": decision}



def retrieve_agent(state: AgentState, retriever_instance: Any) -> dict:
    st.session_state.logs.append("---RETRIEVAL AGENT---")
    query = state.current_query
    try:
        docs_list_objects = retriever_instance.invoke(query)
        retrieved_content_with_meta = []
        for doc in docs_list_objects:
            retrieved_content_with_meta.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
        st.session_state.logs.append(f"Retrieved {len(retrieved_content_with_meta)} documents")
        return {"retrieved_docs": retrieved_content_with_meta}
    except Exception as e:
        st.session_state.logs.append(f"Retrieval error: {str(e)}")
        return {"retrieved_docs": []}



def weather_search_agent(state: AgentState, weather_search_tool: Any, temperature: float) -> dict:
    st.session_state.logs.append("---WEATHER SEARCH AGENT---")
    query = state.current_query
    try:
        model = ChatGroq(
            temperature=temperature,
            model_name="openai/gpt-oss-20b",
            groq_api_key=SECRETS["GROQ_API_KEY"]
        )
        res = model.invoke([
            HumanMessage(content=f"""
        You are given a question and must extract the city name from it.
        Respond ONLY with the city name (no extra text). If no city is found, respond with an empty string.
        Question: {query}
        """)
        ])
        city_name = res.content.strip()
        results = weather_search_tool.run(city_name)

        weather_results_with_meta = [{
            "content": results,
            "metadata": {"source": "weather_search"}
        }]

        st.session_state.logs.append(f"Found weather results for: {city_name}")
        return {"retrieved_docs": weather_results_with_meta}
    except Exception as e:
        st.session_state.logs.append(f"weather search error: {str(e)}")
        return {"retrieved_docs": []}


def generate_agent(state: AgentState , temperature: float) -> dict:
    st.session_state.logs.append("---GENERATION AGENT---")
    if not state.retrieved_docs:
        st.session_state.logs.append("No context available for generation.")
        return {"generated_answer": "I don't have enough information to answer that question."}

    model = ChatGroq(
        temperature= temperature,
        model_name="openai/gpt-oss-20b",
        groq_api_key=SECRETS["GROQ_API_KEY"]
    )

    prompt = hub.pull("rlm/rag-prompt")
    rag_chain = prompt | model | StrOutputParser()

    context_content = "".join([doc["content"] for doc in state.retrieved_docs])

    response = rag_chain.invoke({
        "context": context_content,
        "question": state.current_query
    })

    st.session_state.logs.append("Response generated")
    return {"generated_answer": response}


def route_decision(state: AgentState) -> str:
    return state.next_step