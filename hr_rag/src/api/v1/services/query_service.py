"""
query_service.py
----------------
Service layer — delegates retrieval to the LangChain RAG agent.
Keeps route handlers thin and business logic centralised here.
"""

from src.api.v1.agents.retrieval_rag_agent import run_rag_agent
from src.api.v1.schemas.query_schema import RAGResponse


def handle_query(query: str) -> RAGResponse:
    """
    Handle a user query end-to-end:
      - Invokes the LangChain ReAct agent
      - Agent chooses the retrieval tool (vector / fts / hybrid)
      - Agent synthesises an answer from retrieved chunks
      - Returns a structured RAGResponse

    Args:
        query: The user's natural-language question.

    Returns:
        RAGResponse with query, answer, citation, page_no, document_name.
    """
    return run_rag_agent(query)
