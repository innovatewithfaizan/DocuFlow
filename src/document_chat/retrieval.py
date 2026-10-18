import sys
import os
from operator import itemgetter
from typing import List, Optional, Dict, Any

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
from langchain_community.vectorstores import FAISS

from utils.model_loader import ModelLoader
from utils.file_io import original_name
from exception.custom_exception import DocuFlowException
from logger.custom_logger import CustomLogger
log = CustomLogger().get_logger(__name__)
from prompt.prompt_library import PROMPT_REGISTRY
from model.models import PromptType
from src.document_chat.retriever_factory import RetrieverSettings, build_retriever


class ConversationalRAG:
    """
    LCEL-based Conversational RAG with lazy retriever initialization.

    Usage:
        rag = ConversationalRAG(session_id="abc")
        rag.load_retriever_from_faiss(index_path="faiss_index/abc", k=5, search_type="mmr")
        answer = rag.invoke("What is ...?", chat_history=[])
        result = rag.invoke_with_sources("What is ...?")  # {"answer", "question", "sources", "contexts"}
    """

    def __init__(self, session_id: Optional[str], retriever=None):
        try:
            self.session_id = session_id

            # Load LLM and prompts once
            self.llm = self._load_llm()
            self.contextualize_prompt: ChatPromptTemplate = PROMPT_REGISTRY[
                PromptType.CONTEXTUALIZE_QUESTION.value
            ]
            self.qa_prompt: ChatPromptTemplate = PROMPT_REGISTRY[
                PromptType.CONTEXT_QA.value
            ]

            # Lazy pieces
            self.retriever = retriever
            self.chain = None
            if self.retriever is not None:
                self._build_lcel_chain()

            log.info("ConversationalRAG initialized", session_id=self.session_id)
        except Exception as e:
            log.error("Failed to initialize ConversationalRAG", error=str(e))
            raise DocuFlowException("Initialization error in ConversationalRAG", sys)

    # ---------- Public API ----------

    def load_retriever_from_faiss(
        self,
        index_path: str,
        k: Optional[int] = None,
        index_name: str = "index",
        search_type: Optional[str] = None,
        compress: Optional[bool] = None,
    ):
        """
        Load FAISS vectorstore from disk and build retriever + LCEL chain.
        Unset retrieval options fall back to config.yaml's `retriever:` block.
        """
        try:
            if not os.path.isdir(index_path):
                raise FileNotFoundError(f"FAISS index directory not found: {index_path}")

            embeddings = ModelLoader().load_embeddings()
            vectorstore = FAISS.load_local(
                index_path,
                embeddings,
                index_name=index_name,
                allow_dangerous_deserialization=True,  # only indexes this service wrote itself
            )

            settings = RetrieverSettings.from_config(k=k, search_type=search_type, compress=compress)
            self.settings = settings
            self.retriever = build_retriever(vectorstore, settings, embeddings=embeddings)
            self._build_lcel_chain()

            log.info(
                "FAISS retriever loaded successfully",
                index_path=index_path,
                index_name=index_name,
                search_type=settings.search_type,
                k=settings.k,
                compress=settings.compression.enabled,
                session_id=self.session_id,
            )
            return self.retriever

        except Exception as e:
            log.error("Failed to load retriever from FAISS", error=str(e))
            raise DocuFlowException("Loading error in ConversationalRAG", sys)

    def invoke(self, user_input: str, chat_history: Optional[List[BaseMessage]] = None) -> str:
        """Answer a question; returns only the answer text."""
        return self.invoke_with_sources(user_input, chat_history)["answer"]

    def invoke_with_sources(
        self, user_input: str, chat_history: Optional[List[BaseMessage]] = None
    ) -> Dict[str, Any]:
        """Answer a question and return what it was based on."""
        try:
            if self.chain is None:
                raise DocuFlowException(
                    "RAG chain not initialized. Call load_retriever_from_faiss() before invoke().", sys
                )
            chat_history = chat_history or []
            result = self.chain.invoke({"input": user_input, "chat_history": chat_history})
            answer = result["answer"]
            if not answer:
                log.warning(
                    "No answer generated", user_input=user_input, session_id=self.session_id
                )
                answer = "no answer generated."
            log.info(
                "Chain invoked successfully",
                session_id=self.session_id,
                user_input=user_input,
                retrieved=len(result["docs"]),
                answer_preview=str(answer)[:150],
            )
            return {
                "answer": answer,
                "question": result["question"],
                "sources": [self._source_info(d) for d in result["docs"]],
                "contexts": [d.page_content for d in result["docs"]],
            }
        except Exception as e:
            log.error("Failed to invoke ConversationalRAG", error=str(e))
            raise DocuFlowException("Invocation error in ConversationalRAG", sys)

    # ---------- Internals ----------

    def _load_llm(self):
        try:
            llm = ModelLoader().load_llm()
            if not llm:
                raise ValueError("LLM could not be loaded")
            log.info("LLM loaded successfully", session_id=self.session_id)
            return llm
        except Exception as e:
            log.error("Failed to load LLM", error=str(e))
            raise DocuFlowException("LLM loading error in ConversationalRAG", sys)

    @staticmethod
    def _format_docs(docs) -> str:
        return "\n\n".join(getattr(d, "page_content", str(d)) for d in docs)

    @staticmethod
    def _source_info(doc: Document) -> Dict[str, Any]:
        md = doc.metadata or {}
        page = md.get("page")
        return {
            "source": original_name(str(md.get("source", "unknown"))),
            "page": page + 1 if isinstance(page, int) else None,  # loaders count from 0
            "snippet": doc.page_content[:200],
        }

    def _build_lcel_chain(self):
        try:
            if self.retriever is None:
                raise DocuFlowException("No retriever set before building chain", sys)

            # 1) Standalone question. Rewriting only helps follow-ups, so a first
            #    question goes to the retriever as-is (saves one LLM call per query).
            rewrite = self.contextualize_prompt | self.llm | StrOutputParser()
            standalone_question = RunnableBranch(
                (lambda x: not x.get("chat_history"), itemgetter("input")),
                rewrite,
            )

            # 2) Retrieve for the standalone question, 3) answer from that context
            answer = self.qa_prompt | self.llm | StrOutputParser()
            self.chain = (
                RunnablePassthrough.assign(question=standalone_question)
                | RunnablePassthrough.assign(docs=itemgetter("question") | self.retriever)
                | RunnablePassthrough.assign(context=lambda x: self._format_docs(x["docs"]))
                | RunnablePassthrough.assign(answer=answer)
            )

            log.info("LCEL graph built successfully", session_id=self.session_id)
        except Exception as e:
            log.error("Failed to build LCEL chain", error=str(e), session_id=self.session_id)
            raise DocuFlowException("Failed to build LCEL chain", sys)
