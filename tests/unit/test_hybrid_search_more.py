import pytest

import hybrid_search

pytestmark = pytest.mark.unit


def test_reciprocal_rank_fusion_scores():
    list_a = [{"id": "d1", "document": "a"}]
    list_b = [{"id": "d1", "document": "a"}]
    fused = hybrid_search.reciprocal_rank_fusion([list_a, list_b], k=1, rrf_k=60)
    assert fused[0]["id"] == "d1"
    assert fused[0]["rrf_score"] > 0
