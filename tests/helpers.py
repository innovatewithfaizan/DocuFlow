"""
Test doubles shared by the suite. Nothing here touches the network or needs API keys:

- HashingEmbeddings: deterministic bag-of-words vectors, so texts that share words
  are close. Retrieval tests can therefore assert *which* chunk comes back.
- RecordingChatModel: a chat model that records every prompt it receives and
  returns a fixed answer (or echoes the question for the rewrite step).
"""
from __future__ import annotations

import re
import zipfile
import zlib
from pathlib import Path
from typing import Any, List, Optional
from xml.sax.saxutils import escape

import fitz  # PyMuPDF
import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field


REWRITE_PROMPT_PREFIX = "Given a conversation history"


class HashingEmbeddings(Embeddings):
    def __init__(self, dim: int = 512):
        self.dim = dim

    def _embed(self, text: str) -> List[float]:
        vec = np.zeros(self.dim)
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            vec[zlib.crc32(token.encode()) % self.dim] += 1.0
        norm = np.linalg.norm(vec)
        if norm == 0:
            vec[0], norm = 1.0, 1.0
        return (vec / norm).tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


class RecordingChatModel(BaseChatModel):
    answer: str = "stub answer"
    prompts: List[List[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "recording-fake"

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None,
                  run_manager: Any = None, **kwargs: Any) -> ChatResult:
        self.prompts.append(messages)
        system = messages[0].content if messages and messages[0].type == "system" else ""
        # Question-rewrite step: return the question unchanged
        text = messages[-1].content if str(system).startswith(REWRITE_PROMPT_PREFIX) else self.answer
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    def qa_contexts(self) -> List[str]:
        """System messages of the answer prompts, i.e. the context the LLM was given."""
        return [str(p[0].content) for p in self.prompts
                if p and p[0].type == "system" and not str(p[0].content).startswith(REWRITE_PROMPT_PREFIX)]


class FakeUpload:
    """Mimics the objects the ingestion layer receives (`.name` + `.getbuffer()`)."""

    def __init__(self, path: Path, name: Optional[str] = None):
        self.name = name or path.name
        self._data = path.read_bytes()

    def getbuffer(self) -> bytes:
        return self._data


def make_pdf(path: Path, pages: List[str]) -> Path:
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(72, 72, 540, 770), text, fontsize=10)
    doc.save(str(path))
    doc.close()
    return path


def make_docx(path: Path, paragraphs: List[str]) -> Path:
    body = "".join(f"<w:p><w:r><w:t>{escape(p)}</w:t></w:r></w:p>" for p in paragraphs)
    xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{body}</w:body></w:document>"
    )
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("word/document.xml", xml)
    return path
