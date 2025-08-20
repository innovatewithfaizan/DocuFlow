from __future__ import annotations
import os
import sys
import json
import uuid
import hashlib
import shutil
from pathlib import Path
from typing import Iterable, List, Optional, Dict, Any
import fitz  # PyMuPDF
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from utils.model_loader import ModelLoader
from logger.custom_logger import CustomLogger as log
from exception.custom_exception import DocumentPortalException
from utils.file_io import generate_session_id, save_uploaded_files
from utils.document_ops import load_documents, concat_for_analysis, concat_for_comparison

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


class FaissManager:
    def __init__(self):
        pass
    
    def _exists(self):
        pass

    @staticmethod
    def _fingerprint(self):
        pass

    def _save_meta(self):
        pass
    
    def add_documents(self):
        pass

    def load_or_create(self):
        pass


class DocHandler:
    def __init__(self):
        pass

    def save_pdf(self):
        pass
    
    def read_pdf(self):
        pass

class DocumentComparator:
    def __init__(self):
        pass

    def save_uploaded_files(self):
        pass

    def read_pdf(self):
        pass

    def combine_documents(self):
        pass

    def clean_old_sessions(self):
        pass

class ChatIngestor:
    def __init__(self):
        pass

    def _resolve_dir(self):
        pass

    def _split(self):
        pass

    def built_retriever(self):
        pass

    