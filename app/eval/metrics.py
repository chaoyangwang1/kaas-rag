"""检索评估模块 - 指标计算"""
import math
from typing import List, Set


def compute_all_metrics(relevant_ids: Set[str], retrieved_ids: List[str], k: int = 5) -> dict:
    """计算 Recall@K, Precision@K, MRR, NDCG@K, MAP, Hit Rate"""
    relevant = set(relevant_ids)
    top_k = retrieved_ids[:k]
    hits = [1 if rid in relevant else 0 for rid in top_k]
    hit_count = sum(hits)
    total_relevant = len(relevant) if len(relevant) > 0 else 1

    recall = hit_count / total_relevant
    precision = hit_count / k if k > 0 else 0.0

    # MRR
    mrr = 0.0
    for i, rid in enumerate(top_k):
        if rid in relevant:
            mrr = 1.0 / (i + 1)
            break

    # NDCG@K
    dcg = 0.0
    for i, h in enumerate(hits):
        if h:
            dcg += 1.0 / math.log2(i + 2)
    ideal_count = min(total_relevant, k)
    ideal_hits = [1] * ideal_count + [0] * (k - ideal_count)
    idcg = 0.0
    for i, h in enumerate(ideal_hits):
        if h:
            idcg += 1.0 / math.log2(i + 2)
    ndcg = dcg / idcg if idcg > 0 else 0.0

    # MAP
    sum_precision = 0.0
    running_hits = 0
    for i, h in enumerate(hits):
        if h:
            running_hits += 1
            sum_precision += running_hits / (i + 1)
    ap = sum_precision / total_relevant if total_relevant > 0 else 0.0

    # Hit Rate
    hit_rate = 1.0 if hit_count > 0 else 0.0

    return {
        f"recall@{k}": round(recall, 4),
        f"precision@{k}": round(precision, 4),
        "mrr": round(mrr, 4),
        f"ndcg@{k}": round(ndcg, 4),
        "map": round(ap, 4),
        "hit_rate": round(hit_rate, 4),
    }
