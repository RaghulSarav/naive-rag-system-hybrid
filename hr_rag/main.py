"""
main.py
-------
FastAPI application entry-point for the Hybrid RAG System.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.v1.routes.query import router as query_router

app = FastAPI(
    title="HR RAG API",
    description=(
        "Agentic Hybrid RAG system backed by PostgreSQL + PGVector. "
        "A LangChain ReAct agent dynamically selects between vector search, "
        "full-text search, and hybrid search to answer HR knowledge-base queries."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow all origins during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
def read_root():
    return {
        "message": "HR RAG API is running.",
        "docs": "/docs",
        "query_endpoint": "/api/v1/query",
    }


@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok"}


# Mount versioned router
app.include_router(query_router, prefix="/api/v1")
