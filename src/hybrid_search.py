from config import HYBRID_BM25_K, HYBRID_VECTOR_K, RRF_K
from bm25 import search_bm25
from vectorstore import search_vectorstore


def reciprocal_rank_fusion(results_lists: list[list[dict]], k: int, rrf_k: int = 60) -> list[dict]:
    scores: dict[str, float] = {}
    documents_by_id: dict[str, dict] = {}

    for results in results_lists:
        for rank, item in enumerate(results, start=1):
            doc_id = item["id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
            if doc_id not in documents_by_id:
                documents_by_id[doc_id] = item

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:k]
    fused = []
    for rank, (doc_id, score) in enumerate(ranked, start=1):
        item = dict(documents_by_id[doc_id])
        item["rrf_score"] = score
        item["rank"] = rank
        fused.append(item)
    return fused


def search_hybrid(query: str, k: int = 3) -> list[str]:
    vector_results = search_vectorstore(query, k=HYBRID_VECTOR_K)
    bm25_results = search_bm25(query, k=HYBRID_BM25_K)
    fused = reciprocal_rank_fusion([vector_results, bm25_results], k=k, rrf_k=RRF_K)
    return [item["document"] for item in fused]
