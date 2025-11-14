# tests/conftest.py
import os
import types
import pytest
from types import SimpleNamespace
import tempfile
import shutil

class DummyUploadedFile:
    """Mimics Streamlit's uploaded file object used by load_uploaded_docs."""
    def __init__(self, name: str, content: bytes):
        self.name = name
        self._content = content

    def getvalue(self):
        return self._content

@pytest.fixture
def dummy_txt_upload():
    return DummyUploadedFile("example.txt", b"hello world from test")

@pytest.fixture
def dummy_empty_upload():
    return DummyUploadedFile("empty.txt", b"")

@pytest.fixture
def tmp_default_pdf(tmp_path):
    """
    Create a temporary base dir that mimics your app's `pdf_file/Riyanshu_Resume.pdf`
    Return the base dir path so tests can pass it into initialize_system via default_pdf_path.
    """
    base = tmp_path / "app_base"
    pdf_dir = base / "pdf_file"
    pdf_dir.mkdir(parents=True)
    pdf = pdf_dir / "Riyanshu_Resume.pdf"
    pdf.write_bytes(b"%PDF-1.4 dummy pdf content")
    return str(base)
