"""检索评估模块 - 评估引擎"""
from datetime import datetime
from app.eval.models import EvalTask, EvalResult, PerQueryDetail, StageMetrics, QueryItem
from app.eval.metrics import compute_all_metrics
from app.eval.db import tasks_col, items_col
from app.core.logger import logger


def run_eval_task(task: EvalTask, dataset_items: list[QueryItem]):
    """执行评估任务：遍历参数组 → 遍历问题 → 检索 → 计算指标"""
    task.status = "running"
    tasks_col().update_one(
        {"task_id": task.task_id},
        {"$set": {"status": "running"}}
    )

    all_results = {}

    try:
        for gi, param_group in enumerate(task.param_groups):
            per_query = []
            stage_hits = {"embedding": 0, "hyde": 0, "rrf": 0, "rerank": 0}

            for item in dataset_items:
                detail = _eval_single_query(item, param_group, stage_hits, task.kb_id)
                per_query.append(detail)

            total = len(dataset_items)
            overall = _aggregate_metrics(per_query, param_group.top_k)

            stage = StageMetrics(
                embedding_recall=round(stage_hits["embedding"] / max(total, 1), 4),
                hyde_recall=round(stage_hits["hyde"] / max(total, 1), 4),
                rrf_recall=round(stage_hits["rrf"] / max(total, 1), 4),
                rerank_recall=round(stage_hits["rerank"] / max(total, 1), 4),
            )

            result = EvalResult(
                group_index=gi,
                params=param_group.model_dump(),
                overall_metrics=overall,
                stage_metrics=stage,
                per_query_details=[d.model_dump() for d in per_query],
            )
            all_results[f"group_{gi}"] = result.model_dump()

        task.status = "completed"
    except Exception as e:
        logger.error(f"评估任务失败: {e}", exc_info=True)
        task.status = "failed"

    tasks_col().update_one(
        {"task_id": task.task_id},
        {"$set": {
            "status": task.status,
            "results": all_results,
            "finished_at": datetime.now().isoformat()
        }}
    )


def _eval_single_query(item: QueryItem, param_group, stage_hits: dict, kb_id: int = 1) -> PerQueryDetail:
    """对单个问题执行完整检索流程，记录各阶段结果"""
    from app.clients.milvus_utils import get_milvus_client, create_hybrid_search_requests, hybrid_search
    from app.lm.embedding_utils import generate_embeddings

    relevant = set(item.relevant_chunk_ids)
    client = get_milvus_client()
    if not client:
        logger.warning(f"评估: Milvus 客户端不可用，跳过问题 [{item.question[:30]}...]")
        return PerQueryDetail(
            question=item.question,
            relevant_ids=list(relevant),
            retrieved_ids=[],
            hit=False,
        )

    # 生成向量
    embeddings = generate_embeddings([item.question])
    dense_vec = embeddings.get("dense", [[]])[0]
    sparse_vec = embeddings.get("sparse", [[]])[0]

    if not dense_vec:
        logger.warning(f"评估: 向量生成为空，跳过问题 [{item.question[:30]}...]")
        return PerQueryDetail(
            question=item.question,
            relevant_ids=list(relevant),
            retrieved_ids=[],
            hit=False,
        )

    # 构造过滤表达式：按 kb_id 过滤 + 按 item_name 过滤（若有）
    expr = f"kb_id == {kb_id}"
    if item.item_name:
        safe_name = item.item_name.replace('"', '\\"')
        expr += f' and item_name == "{safe_name}"'
    logger.debug(f"评估检索 expr: {expr}")

    reqs = create_hybrid_search_requests(
        dense_vec, sparse_vec,
        expr=expr,
        limit=param_group.top_k * 3,
    )
    weights = (param_group.dense_weight, param_group.sparse_weight)

    # 阶段1: embedding 混合检索
    res = hybrid_search(
        client=client,
        collection_name=param_group.collection_name,
        reqs=reqs,
        ranker_weights=weights,
        limit=param_group.top_k,
        output_fields=["chunk_id"],
    )
    emb_ids = _extract_chunk_ids(res)
    emb_hit = any(rid in relevant for rid in emb_ids[:param_group.top_k])
    if emb_hit:
        stage_hits["embedding"] += 1

    # 阶段2: HyDE（与 embedding 共享相同的 dense+sparse 向量，这里简化处理）
    stage_hits["hyde"] += (1 if emb_hit else 0)

    # 阶段3: RRF 融合后（此处简化，使用嵌入结果近似）
    stage_hits["rrf"] += (1 if emb_hit else 0)

    # 阶段4: Rerank 重排后
    rerank_ids = emb_ids[:param_group.top_k]
    if any(rid in relevant for rid in rerank_ids):
        stage_hits["rerank"] += 1

    hit = any(rid in relevant for rid in rerank_ids)
    rank = None
    for i, rid in enumerate(rerank_ids):
        if rid in relevant:
            rank = i + 1
            break

    logger.debug(f"评估: [{item.question[:30]}...] 相关={len(relevant)} 检索={len(rerank_ids)} 命中={hit}")

    return PerQueryDetail(
        question=item.question,
        relevant_ids=list(relevant),
        retrieved_ids=rerank_ids,
        rank=rank,
        hit=hit,
    )


def _extract_chunk_ids(search_result) -> list[str]:
    """从 Milvus 搜索结果中提取 chunk_id 列表（chunk_id 是 auto_id 主键，存在 hit['id']）"""
    ids = []
    if search_result and len(search_result) > 0:
        for hit in search_result[0]:
            cid = hit.get("id")
            if cid is not None:
                ids.append(str(cid))
    return ids


def _aggregate_metrics(per_query: list[PerQueryDetail], k: int) -> dict:
    """汇总所有问题的指标，取平均值"""
    key_metrics = ["mrr", f"recall@{k}", f"precision@{k}", f"ndcg@{k}", "map", "hit_rate"]
    sums = {m: 0.0 for m in key_metrics}
    for detail in per_query:
        metrics = compute_all_metrics(set(detail.relevant_ids), detail.retrieved_ids, k=k)
        for m in key_metrics:
            sums[m] += metrics.get(m, 0.0)
    n = max(len(per_query), 1)
    return {m: round(sums[m] / n, 4) for m in key_metrics}
