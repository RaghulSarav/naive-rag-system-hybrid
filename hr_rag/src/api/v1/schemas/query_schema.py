"""
query_schema.py
---------------
Pydantic schemas for the RAG query endpoint.
"""

from typing import Optional
from pydantic import BaseModel, Field


# ---- Request ----
class QueryRequest(BaseModel):
    query: str = Field(..., description="The user's question to the HR knowledge base.")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"query": "What is the leave encashment policy?"},
                {"query": "POL-2024-HR-007"},
                {"query": "LTA rules"},
            ]
        }
    }


# ---- Response ----
class RAGResponse(BaseModel):
    query: str = Field(..., description="The original user query.")
    answer: str = Field(..., description="Synthesised answer from retrieved chunks.")
    citation: str = Field(..., description="Verbatim excerpt from the source document.")
    page_no: Optional[int] = Field(
        default=None, description="Page number in the source PDF (0-indexed)."
    )
    document_name: str = Field(
        ..., description="Filename of the source document."
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "What is the leave encashment policy?",
                    "answer": "Employees can encash up to 30 days of earned leave per year.",
                    "citation": "Unused earned leave may be encashed at the rate of basic pay...",
                    "page_no": 4,
                    "document_name": "HR_Support_Desk_KnowledgeBase.pdf",
                }
            ]
        }
    }
