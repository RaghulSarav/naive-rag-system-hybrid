"""
query.py (routes)
-----------------
FastAPI route: POST /api/v1/query

Accepts a QueryRequest and returns a RAGResponse via the query service.
"""

from fastapi import APIRouter, HTTPException
from src.api.v1.schemas.query_schema import QueryRequest, RAGResponse
from src.api.v1.services.query_service import handle_query

router = APIRouter(tags=["Query"])


@router.post(
    "/query",
    response_model=RAGResponse,
    summary="Query the HR knowledge base",
    description=(
        "Send a natural-language question to the RAG agent. "
        "The agent selects the best retrieval strategy (vector, FTS, or hybrid), "
        "fetches relevant chunks, and returns a structured answer with citation "
        "and source metadata."
    ),
)
def query_endpoint(request: QueryRequest) -> RAGResponse:
    """
    RAG query endpoint.

    The LangChain ReAct agent decides whether to call:
    - **vector_search** for conceptual / natural-language questions
    - **fts_search** for keyword codes or exact terms
    - **hybrid_search** for short or ambiguous queries

    Returns a JSON response with:
    - `query`         — the original question
    - `answer`        — synthesised answer
    - `citation`      — verbatim excerpt from the source
    - `page_no`       — page number in the PDF
    - `document_name` — source filename
    """
    try:
        return handle_query(request.query)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
