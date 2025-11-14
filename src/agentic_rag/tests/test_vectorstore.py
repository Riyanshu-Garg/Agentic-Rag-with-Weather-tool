# tests/test_vectorstore.py
import hashlib
import pytest
from unittest.mock import MagicMock, patch

from vectorstore import calculate_knowledge_hash, split_documents, build_qdrant_vectorstore

class DummyUploaded:
    def __init__(self, name, content: bytes):
        self.name = name
        self._content = content
    def getvalue(self):
        return self._content

def test_calculate_knowledge_hash_single_file():
    f = DummyUploaded("a.txt", b"abc123")
    h = calculate_knowledge_hash([f])
    expected = hashlib.md5(f.getvalue().decode(errors="ignore").encode()).hexdigest()
    assert h == expected

@patch("vectorstore.RecursiveCharacterTextSplitter")
def test_split_documents_calls_text_splitter(mock_splitter_class):
    # Prepare fake documents
    docs = [type("D", (), {"page_content": "x" * 1000})()]
    mock_splitter = MagicMock()
    mock_splitter.split_documents.return_value = ["chunk1", "chunk2"]
    mock_splitter_class.from_tiktoken_encoder.return_value = mock_splitter

    res = split_documents(docs, chunk_size=250)
    assert res == ["chunk1", "chunk2"]
    mock_splitter_class.from_tiktoken_encoder.assert_called_once()

def test_build_qdrant_vectorstore_handles_qdrant_recreate_exception(monkeypatch, tmp_path):
    # Simulate a QdrantClient that raises on recreate_collection
    class BadClient:
        def __init__(self, *a, **k):
            pass
        def recreate_collection(self, *a, **k):
            raise RuntimeError("qdrant unavailable")

    monkeypatch.setenv("GOOGLE_API_KEY", "fake")
    monkeypatch.setenv("QDRANT_URL", "fake")
    monkeypatch.setenv("QDRANT_API", "fake")

    monkeypatch.setattr("vectorstore.QdrantClient", lambda url, api_key: BadClient())
    # patch qdrant_client.http.models.VectorParams to a dummy to avoid import errors
    monkeypatch.setattr("vectorstore.qdrant_client.http.models.VectorParams", lambda **k: object())
    monkeypatch.setattr("vectorstore.qdrant_client.http.models.Distance", type("D", (), {"COSINE": "cos"}) )

    # Calling build_qdrant_vectorstore should raise the runtime error from recreate_collection
    with pytest.raises(RuntimeError):
        build_qdrant_vectorstore([], google_api_key="g", qdrant_url="u", qdrant_api="a")
