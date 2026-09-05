from app.rag_pipeline import build_regulation_chunks, retrieve_relevant_chunks


def test_build_regulation_chunks_creates_clause_like_entries() -> None:
    chunks = build_regulation_chunks()

    assert len(chunks) >= 200
    assert all("regulation_number" in chunk for chunk in chunks)
    assert all("title" in chunk for chunk in chunks)
    assert all("content" in chunk for chunk in chunks)
    assert any("Exposure" in chunk["title"] or "Exposure" in chunk["content"] for chunk in chunks)


def test_retrieve_relevant_chunks_returns_ranked_matches() -> None:
    chunks = build_regulation_chunks()
    matches = retrieve_relevant_chunks("single borrower exposure limit and sector concentration", chunks)

    assert len(matches) >= 3
    assert matches[0]["regulation_number"] in {"R-5", "R-7", "R-12"}
    assert all("score" in result for result in matches)
