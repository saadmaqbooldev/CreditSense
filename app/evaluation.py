from __future__ import annotations

from typing import Any
from app.rag_pipeline import build_regulation_chunks, retrieve_relevant_chunks


class RetrievalEvaluator:
    """Evaluate retrieval quality using precision@k and recall@k metrics."""
    
    def __init__(self) -> None:
        self.chunks = build_regulation_chunks()
    
    def evaluate_retrieval(
        self,
        query: str,
        expected_regulation_numbers: list[str],
        k: int = 5,
    ) -> dict[str, Any]:
        """
        Evaluate retrieval quality.
        
        Args:
            query: Search query
            expected_regulation_numbers: List of regulation numbers that should be retrieved
            k: Number of results to evaluate (default 5)
        
        Returns:
            Metrics: precision@k, recall@k, retrieved regulations, expected regulations
        """
        retrieved = retrieve_relevant_chunks(query, self.chunks)[:k]
        retrieved_ids = [chunk["regulation_number"] for chunk in retrieved]
        
        if not expected_regulation_numbers:
            precision = 1.0 if not retrieved_ids else 0.0
            recall = 1.0
        else:
            relevant_retrieved = set(retrieved_ids) & set(expected_regulation_numbers)
            precision = len(relevant_retrieved) / len(retrieved_ids) if retrieved_ids else 0.0
            recall = len(relevant_retrieved) / len(expected_regulation_numbers)
        
        return {
            "query": query,
            "retrieved_regulations": retrieved_ids,
            "expected_regulations": expected_regulation_numbers,
            "precision_at_k": round(precision, 4),
            "recall_at_k": round(recall, 4),
            "k": k,
        }
    
    def evaluate_batch(
        self,
        test_cases: list[dict[str, Any]],
        k: int = 5,
    ) -> dict[str, Any]:
        """
        Evaluate a batch of retrieval test cases.
        
        Args:
            test_cases: List of {query, expected_regulation_numbers}
            k: Number of results per query
        
        Returns:
            Summary metrics and per-query results
        """
        results = []
        precision_scores = []
        recall_scores = []
        
        for case in test_cases:
            result = self.evaluate_retrieval(case["query"], case.get("expected_regulation_numbers", []), k)
            results.append(result)
            precision_scores.append(result["precision_at_k"])
            recall_scores.append(result["recall_at_k"])
        
        avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0.0
        avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
        
        return {
            "test_count": len(test_cases),
            "avg_precision_at_k": round(avg_precision, 4),
            "avg_recall_at_k": round(avg_recall, 4),
            "k": k,
            "results": results,
        }


RETRIEVAL_TEST_CASES = [
    {
        "query": "single borrower exposure limit",
        "expected_regulation_numbers": ["R-5"],
    },
    {
        "query": "sector concentration risk diversification",
        "expected_regulation_numbers": ["R-7"],
    },
    {
        "query": "collateral valuation standards",
        "expected_regulation_numbers": ["R-9"],
    },
    {
        "query": "leverage debt equity ratio",
        "expected_regulation_numbers": ["R-12"],
    },
    {
        "query": "late payment delinquency reporting",
        "expected_regulation_numbers": ["R-14"],
    },
    {
        "query": "cash flow assessment SME",
        "expected_regulation_numbers": ["R-18"],
    },
    {
        "query": "underwriting governance auditability",
        "expected_regulation_numbers": ["R-21"],
    },
    {
        "query": "exposure limits and borrower concentration",
        "expected_regulation_numbers": ["R-5"],
    },
    {
        "query": "sectoral exposure and risk",
        "expected_regulation_numbers": ["R-7"],
    },
    {
        "query": "maximum leverage and debt controls",
        "expected_regulation_numbers": ["R-12"],
    },
]
