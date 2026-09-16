import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

from config import GROQ_MODEL

_PROMPT = """\
You are a research assistant with access to a corpus of academic papers from \
networking and systems conferences (SIGCOMM, IMC, CoNEXT, INFOCOM, MobiCom, \
MobiSys, EuroSys, ICDCS), covering 2022-2026.

A researcher has asked the following question:
"{query}"

Here are the most relevant excerpts from the corpus:

{context}

Using ONLY the excerpts above, write a concise answer to the question. \
Write in your own words. Do not quote the excerpts directly. \
For every claim you make, cite the paper it comes from using [Paper Title, \
Conference YEAR] format. If the excerpts do not contain enough information \
to answer the question, say so explicitly — do not speculate or draw on \
outside knowledge.

Answer:
"""

def _build_llm() -> ChatGroq:
    return ChatGroq(
        model=GROQ_MODEL,
        temperature=0,
        max_tokens=8192,
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
            grounded — True only if the model actually produced an answer.
                       False when there were no chunks to ground against or when
                       generation failed or came back empty.
                       Callers (including the eval, via adapters.generate_answer)
                       treat grounded=True as "an answer was produced"; a failure
                       reported as True is counted as a real answer and scored.
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
        answer   = (response.content or "").strip()
    except Exception as e:
        return {
            "answer":   f"(Generation failed: {e})",
            "sources":  [],
            "grounded": False,
        }

    if not answer:
        return {
            "answer":   "(Generation returned no content.)",
            "sources":  [],
            "grounded": False,
        }

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
