import os
import streamlit as st
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
from langchain_core.messages import BaseMessage

def load_secrets_from_streamlit():
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_API_KEY"] = st.secrets.get("LANGCHAIN_API_KEY", "")
    os.environ["TAVILY_API_KEY"] = st.secrets.get("TAVILY_API_KEY", "")
    os.environ["OPENWEATHERMAP_API_KEY"] = st.secrets.get("OPEN_WEATHER_API_KEY", "")


    secrets = {
    "GOOGLE_API_KEY": st.secrets.get("GOOGLE_API_KEY", ""),
    "GROQ_API_KEY": st.secrets.get("GROQ_API_KEY", ""),
    "QDRANT_API": st.secrets.get("Qdrant_API_KEY", ""),
    "QDRANT_URL": st.secrets.get("Qdrant_END_POINT", ""),
    }
    return secrets

SECRETS = load_secrets_from_streamlit()

class AgentState(BaseModel):
    messages: List[BaseMessage] = Field(default_factory=list)
    chat_history: List[BaseMessage] = Field(default_factory=list)
    current_query: Optional[str] = None
    retrieved_docs: List[Dict[str, Any]] = Field(default_factory=list)
    generated_answer: Optional[str] = None
    next_step: Optional[str] = None


    model_config = ConfigDict(arbitrary_types_allowed=True)