"""检索评估指标计算 - 单元测试"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.eval.metrics import compute_all_metrics


class TestRecallAtK:
    def test_partial_hit(self):
        result = compute_all_metrics({"c1", "c2", "c3"}, ["c1", "c2", "c4", "c5"], k=3)
        assert result["recall@3"] == 2 / 3

    def test_no_hit(self):
        result = compute_all_metrics({"c1"}, ["c2", "c3"], k=3)
        assert result["recall@3"] == 0.0
        assert result["mrr"] == 0.0


class TestMRR:
    def test_mrr_first_place(self):
        result = compute_all_metrics({"c1"}, ["c1", "c2", "c3"], k=5)
        assert result["mrr"] == 1.0

    def test_mrr_third_place(self):
        result = compute_all_metrics({"c1"}, ["c2", "c3", "c1"], k=5)
        assert result["mrr"] == 1 / 3


class TestNDCG:
    def test_perfect_ndcg(self):
        result = compute_all_metrics({"c1", "c2"}, ["c1", "c2", "c3"], k=3)
        assert result["ndcg@3"] == 1.0

    def test_partial_ndcg(self):
        result = compute_all_metrics({"c1", "c2"}, ["c1", "c3", "c4"], k=3)
        assert 0 < result["ndcg@3"] < 1.0


class TestPrecision:
    def test_precision(self):
        result = compute_all_metrics({"c1", "c2"}, ["c1", "c3", "c4"], k=3)
        assert result["precision@3"] == 1 / 3


class TestHitRate:
    def test_hit(self):
        result = compute_all_metrics({"c1"}, ["c2", "c1", "c3"], k=3)
        assert result["hit_rate"] == 1.0

    def test_miss(self):
        result = compute_all_metrics({"c1"}, ["c2", "c3", "c4"], k=3)
        assert result["hit_rate"] == 0.0
