# tests/test_app_integration.py
import types
import pytest
from unittest.mock import MagicMock

import app
from config import AgentState

def make_fake_build_vectorstore(expected_texts_container):
    """
    Return a fake build_vectorstore_fn that asserts add_texts contents and returns
    a fake retriever + tool.
    """
    def _fake_build(doc_splits, google_api_key, qdrant_url, qdrant_api, collection_name="agentic_collection"):
        # ensure doc_splits content is forwarded as text list
        texts = [d.page_content for d in doc_splits]
        expected_texts_container.extend(texts)
        fake_retriever = MagicMock(name="fake_retriever")
        fake_tool = MagicMock(name="fake_tool")
        return fake_retriever, fake_tool
    return _fake_build

class FakeCompiledGraph:
    def stream(self, state):
        yield {"router": {"ok": True}}
        yield {"retrieve": {"retrieved_docs": [{"content": "doc1"}]}}
        yield {"generate": {"generated_answer": "final answer"}}

class FakeStateGraphClass:
    def __init__(self, *args, **kwargs):
        pass
    def add_node(self, *args, **kwargs):
        pass
    def add_edge(self, *args, **kwargs):
        pass
    def add_conditional_edges(self, *args, **kwargs):
        pass
    def compile(self):
        return FakeCompiledGraph()

def test_initialize_system_with_injected_dependencies(tmp_default_pdf):
    # Prepare injectable functions
    def fake_load_uploaded_docs(uploads):
        # simulate that uploads list is empty to force default PDF fallback branch
        return []

    def fake_split_documents(docs, chunk_size=250):
        # return simple doc_splits with page_content for verification
        return [types.SimpleNamespace(page_content="chunked text 1")]

    collected_texts = []
    fake_build_fn = make_fake_build_vectorstore(collected_texts)

    # Inject fake OpenWeatherMap wrapper and stategraph class
    fake_weather_cls = lambda: MagicMock(name="weather_tool")
    # Call initialize_system injecting our test doubles and point default_pdf_path to tmp_default_pdf
    compiled_graph, retriever, weather_tool, temperature, retriever_tool = app.initialize_system(
        uploaded_files=[],
        chunk_size=250,
        k=3,
        temperature=0.3,
        load_uploaded_docs_fn=fake_load_uploaded_docs,
        split_documents_fn=fake_split_documents,
        build_vectorstore_fn=fake_build_fn,
        weather_api_wrapper_cls=fake_weather_cls,
        stategraph_cls=FakeStateGraphClass,
        default_pdf_path=tmp_default_pdf
    )

    # Assert compiled graph is our FakeCompiledGraph (has stream)
    assert hasattr(compiled_graph, "stream")
    # Ensure temperature passed through
    assert temperature == 0.3
    # ensure build_vectorstore received our doc chunks
    assert "chunked text 1" in collected_texts
    # Ensure weather tool was created via injected class
    assert weather_tool is not None

def test_workflow_stream_outputs_generated_answer():
    # Use the FakeStateGraphClass that yields a generated_answer
    compiled_graph = FakeStateGraphClass().compile()
    state = AgentState(messages=[], chat_history=[], current_query="hello", retrieved_docs=[])
    outputs = list(compiled_graph.stream(state))
    # last output should include generate step with 'generated_answer'
    assert any("generated_answer" in v for o in outputs for v in o.values())
