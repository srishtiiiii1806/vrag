import os
from pathlib import Path
from typing import List, Literal
from functools import lru_cache
from fastapi import HTTPException, Field
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama.llms import OllamaLLM
from langchain_openai import ChatOpenAI
from loguru import logger

from rag.bm25_retriever import load_bm25_retriever
from rag.prompt import contextualize_q_prompt, qa_prompt
from utils.config import settings
from utils.helper import load_agent_config
from utils.models import Query

os.environ["OPENAI_API_KEY"] = "ollama"

# -------- Global Configuration --------
AGENTS_DIR = Path(settings.agent_dir)


class ModelComponents:
    """Vectorless RAG components — retrieval is done via BM25 over stored
    chunk files instead of a FAISS vector store, but the chain-building and
    chunk-retrieval API mirrors the embedding-based version 1:1."""

    def initialize_llm(self, llm_name: str):
        """Initialize remote LLM once at startup"""
        return OllamaLLM(
            model=llm_name,
            num_predict=1024,
            temperature=0.1,
        )

    def get_rag_chain(
        self,
        db_path: str,
        num_retrieval: int = Field(default=3, ge=1, le=5),
        model_name: str = settings.llm_name,
    ):
        """Create new RAG chain per request, backed by a BM25 chunk retriever"""
        chunks_path = os.path.join(db_path, "chunks")
        if not os.path.exists(chunks_path):
            raise HTTPException(
                status_code=404, detail=f"Chunk store not found at {chunks_path}"
            )

        retriever = get_cached_bm25(
            db_path=chunks_path,
            k=num_retrieval,
        )

        llm = self.initialize_llm(model_name)
        logger.info("Successfully initialized LLM")

        history_aware_retriever = create_history_aware_retriever(
            llm=llm, retriever=retriever, prompt=contextualize_q_prompt
        )

        question_answer_chain = create_stuff_documents_chain(llm=llm, prompt=qa_prompt)

        return create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def get_relevant_chunks(
        self,
        query: str,
        db_path: str,
        num_retrieval: int = 3,
        method: Literal["basic", "mqr"] = "basic",
        enable_rerank: bool = False,
    ):
        """Retrieve relevant chunks for a given query using BM25 (vectorless) retrieval"""
        chunks_path = os.path.join(db_path, "chunks")
        if not os.path.exists(chunks_path):
            raise HTTPException(
                status_code=404, detail=f"Chunk store not found at {chunks_path}"
            )

        if method == "basic":
            retriever = get_cached_bm25(db_path=chunks_path, k=num_retrieval)
            relevant_docs = retriever.invoke(query)
            logger.info(relevant_docs)
        elif method == "mqr":
            retrieval_k = 10 * num_retrieval if enable_rerank else num_retrieval
            base_retriever = get_cached_bm25(
                db_path=chunks_path,
                k=retrieval_k,
            )

            llm = ChatOpenAI(model=settings.rr_llm_name, base_url=settings.rr_llm_url)
            multi_query_retriever = MultiQueryRetriever.from_llm(
                retriever=base_retriever,
                llm=llm,
            )
            relevant_docs = multi_query_retriever.invoke(query)
            logger.info(relevant_docs)

            if enable_rerank:
                compressor = FlashrankRerank()
                compression_retriever = ContextualCompressionRetriever(
                    base_compressor=compressor,
                    base_retriever=multi_query_retriever,
                )
                relevant_docs = compression_retriever.invoke(query)
                logger.info(f"Reranked relevant docs: {relevant_docs}")
        return relevant_docs


def reformat_chat(chat_history: List[str]):
    """
    Convert a list of chat history strings to LangChain message objects.
    """
    formatted_history = []
    if not chat_history:
        return chat_history
    for msg in chat_history:
        if ":" not in msg:
            continue

        role, msg_content = msg.split(":", 1)
        role = role.strip()
        msg_content = msg_content.strip()
        if role == "User":
            formatted_history.append(HumanMessage(content=msg_content))
        elif role == "Assistant":
            formatted_history.append(AIMessage(content=msg_content))
    return formatted_history


def handle_greetings_and_thanks(question: str, rfp: bool = False) -> str:
    greetings = {"hi", "hello", "hey"}
    thanks = {"thanks", "thank you", "thank"}
    # for ask-sam responses
    if question.lower().strip() in greetings:
        response = "Hello, Welcome! How can I assist you today?"
    elif question.lower().strip() in thanks:
        response = (
            "You're welcome! If you have any other questions, feel free to ask again."
        )
    else:
        response = ""
    return response


model_components = ModelComponents()

@lru_cache(maxsize=32)
def get_cached_bm25(db_path: str, k: int):
    return load_bm25_retriever(
        db_path=db_path,
        k=k,
    )
def ask_question(query: Query):
    """Request-scoped, vectorless RAG processing"""
    try:
        question = query.question
        response = handle_greetings_and_thanks(question)
        if response != "":
            return response
        if not question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        chat_history = reformat_chat(query.chat_history) or []

        # Fetch model from MongoDB based on tenant_id and agent_name
        config = load_agent_config(query.eRep_id)
        if not config:
            raise HTTPException(
                status_code=404,
                detail=f"Agent with eRep id {query.eRep_id} not found. Please create embeddings first.",
            )

        # Use model from config, fallback to query.model if not set in config
        model_name = config.get("model") or settings.llm_name
        logger.info(f"Using model for RAG: {model_name}")

        agent_dir = AGENTS_DIR / config["tenant_id"] / query.eRep_id
        rag_chain = model_components.get_rag_chain(
            db_path=str(agent_dir), model_name=model_name
        )
        logger.info("Successfully created RAG chain")
        response = rag_chain.invoke(
            {
                "input": query.question,
                "chat_history": chat_history,
                "db_name": config["tenant_id"],
            }
        )

        return response["answer"], response["context"]

    except Exception as e:
        logger.error(f"API error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def fetch_relevant_chunks(tenant_id: str, query: Query, num_retrieval: int = 3):
    """Request-scoped, vectorless RAG chunk retrieval (BM25)"""
    try:
        question = query.question

        agent_dir = AGENTS_DIR / tenant_id / query.eRep_id
        relevant_docs = model_components.get_relevant_chunks(
            query=question,
            db_path=str(agent_dir),
            num_retrieval=num_retrieval,
            method=query.method,
        )
        logger.info("Successfully fetched relevant chunks")
        return [
            {"content": doc.page_content, "metadata": doc.metadata}
            for doc in relevant_docs
        ]

    except Exception as e:
        logger.error(f"API error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))