"""
retrieval_rag_agent.py
-----------------------
LangChain ReAct agent using the modern `create_agent` API (LangChain 1.0+).

The agent:
  1. Receives the user query
  2. Uses the LLM to decide which retrieval tool to call:
       - vector_search  → semantic / natural-language questions
       - fts_search     → keyword codes, abbreviations, exact terms
       - hybrid_search  → ambiguous / short queries
  3. Synthesises an answer from retrieved chunks
  4. Returns a structured RAGResponse:
       {query, answer, citation, page_no, document_name}
"""

from __future__ import annotations

import json
import os
import re

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

from src.api.v1.tools.vector_search_tool import vector_search
from src.api.v1.tools.fts_search_tool import fts_search
from src.api.v1.tools.hybrid_search_tool import hybrid_search
from src.api.v1.schemas.query_schema import RAGResponse

load_dotenv()

# ---------------------------------------------------------------------------
# 1. LLM
# ---------------------------------------------------------------------------
_llm = ChatGoogleGenerativeAI(
    model=os.getenv("GOOGLE_LLM_MODEL", "gemini-2.0-flash"),
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.2,
)

# ---------------------------------------------------------------------------
# 2. Tools
# ---------------------------------------------------------------------------
_tools = [vector_search, fts_search, hybrid_search]

# ---------------------------------------------------------------------------
# 3. System prompt guiding tool selection + structured output format
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """You are an expert HR knowledge-base assistant with access to
three retrieval tools. Your job is to:

1. Analyse the user query and choose the **best retrieval tool**:
   - `vector_search`  — natural-language questions or conceptual queries
                        (e.g. "What is the work-from-home policy?")
   - `fts_search`     — queries with policy codes, ticket IDs, short uppercase
                        abbreviations, or exact keyword matches
                        (e.g. "LTA", "POL-2024-HR-007", "ESI limit")
   - `hybrid_search`  — short (≤ 3 words) or ambiguous queries where both
                        semantic and keyword recall matter

2. Call the chosen tool with the query.

3. Using ONLY the returned chunks, synthesise a concise, accurate answer.

4. After answering, output a **single line** of valid JSON at the end in this
   exact format (no markdown fences, no extra text after it):
   {{"answer": "...", "citation": "...", "page_no": <integer_or_null>, "document_name": "..."}}

   Where:
   - answer        : your synthesised answer (max 3 sentences)
   - citation      : verbatim excerpt from the most relevant chunk
   - page_no       : integer page number from metadata (0-indexed), or null
   - document_name : value of metadata["document_name"]

If the retrieved chunks do not contain enough information, state that clearly.
"""

# ---------------------------------------------------------------------------
# 4. Agent — using langchain.agents.create_agent (LangChain 1.0+)
# ---------------------------------------------------------------------------
_agent = create_agent(
    llm=_llm,
    tools=_tools,
    prompt=_SYSTEM_PROMPT,
)

# ---------------------------------------------------------------------------
# 5. JSON extraction helper
# ---------------------------------------------------------------------------
_JSON_RE = re.compile(
    r'\{[^{}]*"answer"\s*:[^{}]*"citation"\s*:[^{}]*"page_no"\s*:[^{}]*"document_name"\s*:[^{}]*\}',
    re.DOTALL,
)


def _extract_json(text: str) -> dict:
    """Extract the structured JSON block from the agent's final output."""
    match = _JSON_RE.search(text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    # Fallback: wrap raw text
    return {
        "answer": text.strip(),
        "citation": "",
        "page_no": None,
        "document_name": "",
    }


# ---------------------------------------------------------------------------
# 6. Public entry-point
# ---------------------------------------------------------------------------
def run_rag_agent(query: str) -> RAGResponse:
    """
    Run the LangChain RAG agent for a given user query.

    The agent decides which retrieval tool to use, fetches relevant chunks,
    synthesises an answer, and returns a fully structured RAGResponse.

    Args:
        query: The user's natural-language question.

    Returns:
        RAGResponse with query, answer, citation, page_no, document_name.
    """
    result = _agent.invoke({"input": query})

    # create_agent returns {"output": "..."} from the agent's final message
    output: str = result.get("output", "")

    parsed = _extract_json(output)

    return RAGResponse(
        query=query,
        answer=parsed.get("answer", output.strip()),
        citation=parsed.get("citation", ""),
        page_no=parsed.get("page_no"),
        document_name=parsed.get("document_name", ""),
    )
