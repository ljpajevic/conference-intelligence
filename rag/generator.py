import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

OLLAMA_MODEL = "llama3.1:8b"

_PROMPT = """\
You are a research assistant with access to a corpus of academic papers from \
networking and systems conferences (SIGCOMM, IMC, CoNEXT, INFOCOM, MobiCom, \
MobiSys, EuroSys, ICDCS), covering 2023-2025.

A researcher has asked the following question:
"{query}"

Here are the most relevant excerpts from the corpus:

{context}

Using ONLY the excerpts above, write a concise answer to the question. \
For every claim you make, cite the paper it comes from using [Paper Title, \
Conference YEAR] format. If the excerpts do not contain enough information \
to answer the question, say so explicitly — do not speculate or draw on \
outside knowledge.

Answer:
"""


def _build_llm() -> ChatOllama:
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url="http://localhost:11434",
        temperature=0,
        num_ctx=8192,
    )


def _format_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[{i}] {chunk['paper_title']} "
            f"({chunk['conference'].upper()} {chunk['year']})\n"
            f"{chunk['text']}"
        )
    return "\n\n".join(parts)


def generate(query: str, chunks: list[dict]) -> dict:
    """
    Generate a grounded answer from retrieved chunks.

    Args:
        query:  The user's question.
        chunks: Retrieved chunks from the retriever (may be empty).

    Returns:
        Dict with keys:
            answer  — the generated answer string
            sources — deduplicated list of (paper_title, conference, year, doi)
            grounded — True if chunks were available, False if answer is a fallback
    """
    if not chunks:
        return {
            "answer":   "No papers closely related to this question were found in the corpus.",
            "sources":  [],
            "grounded": False,
        }

    context = _format_context(chunks)
    prompt  = _PROMPT.format(query=query, context=context)

    llm = _build_llm()
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        answer   = response.content.strip()
    except Exception as e:
        answer = f"(Generation failed: {e})"

    # deduplicate sources by paper title
    seen    = set()
    sources = []
    for chunk in chunks:
        key = chunk["paper_title"]
        if key not in seen:
            seen.add(key)
            sources.append({
                "paper_title": chunk["paper_title"],
                "conference":  chunk["conference"],
                "year":        chunk["year"],
                "doi":         chunk["doi"],
            })

    return {
        "answer":   answer,
        "sources":  sources,
        "grounded": True,
    }
