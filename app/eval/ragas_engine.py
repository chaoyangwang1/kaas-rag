"""RAGAS 评估引擎 - 端到端 RAG 质量评估（使用 LLM-as-Judge）"""
import os
import asyncio
from datetime import datetime

import numpy as np
from openai import AsyncOpenAI
from dotenv import load_dotenv
from ragas.llms import llm_factory
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    FactualCorrectness,
)

from app.eval.models import EvalTask, EvalResult, QueryItem
from app.eval.db import tasks_col
from app.core.logger import logger

load_dotenv()

# LLM 配置（从环境变量读取，与项目共用）
LLM_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
LLM_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_DEFAULT_MODEL", "gpt-3.5-turbo")


class _LocalBGE3RagasEmbeddings(BaseRagasEmbedding):
    """用本地 BGE-M3 模型提供 RAGAS 所需的 embedding（实现 BaseRagasEmbedding 接口）"""

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        from app.lm.embedding_utils import generate_embeddings
        emb = generate_embeddings(texts)
        return [[float(v) for v in d] for d in emb["dense"]]

    def embed_text(self, text: str, **kwargs) -> list[float]:
        """同步单文本 embedding"""
        return self._embed_sync([text])[0]

    async def aembed_text(self, text: str, **kwargs) -> list[float]:
        """异步单文本 embedding"""
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(None, self._embed_sync, [text])
        return results[0]


def _build_embeddings():
    """构建 RAGAS 所需的 Embeddings 实例（使用本地 BGE-M3）"""
    return _LocalBGE3RagasEmbeddings()


def run_ragas_eval(task: EvalTask, dataset_items: list[QueryItem]):
    """RAGAS 评估：完整检索+生成 → LLM-as-Judge 评估"""
    task.status = "running"
    tasks_col().update_one({"task_id": task.task_id}, {"$set": {"status": "running"}})

    try:
        all_results = asyncio.run(_run_ragas_async(task, dataset_items))
        task.status = "completed"
    except Exception as e:
        logger.error(f"RAGAS 评估任务失败: {e}", exc_info=True)
        task.status = "failed"
        all_results = {}

    tasks_col().update_one(
        {"task_id": task.task_id},
        {"$set": {
            "status": task.status,
            "results": all_results,
            "finished_at": datetime.now().isoformat(),
        }}
    )


async def _run_ragas_async(task: EvalTask, dataset_items: list[QueryItem]) -> dict:
    """异步执行 RAGAS 评估"""
    client = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    llm = llm_factory(LLM_MODEL, client=client)
    embeddings = _build_embeddings()

    # 初始化指标（RAGAS 0.2+ 要求 AnswerRelevancy/FactualCorrectness 传入 embeddings）
    metrics = {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),
        "context_precision": ContextPrecision(llm=llm),
        "context_recall": ContextRecall(llm=llm),
        "factual_correctness": FactualCorrectness(llm=llm, embeddings=embeddings),
    }

    all_results = {}

    for gi, param_group in enumerate(task.param_groups):
        scores_by_metric = {k: [] for k in metrics}
        per_query = []

        for item in dataset_items:
            detail = await _eval_single_ragas(item, param_group, metrics, task.kb_id)
            for k in scores_by_metric:
                scores_by_metric[k].append(detail.get(k, 0))
            per_query.append(detail)

        # 汇总平均
        n = max(len(dataset_items), 1)
        overall = {k: round(sum(v) / n, 4) for k, v in scores_by_metric.items()}

        result_obj = EvalResult(
            group_index=gi,
            params=param_group.model_dump(),
            overall_metrics=overall,
            per_query_details=per_query,
        )
        all_results[f"group_{gi}"] = result_obj.model_dump()

    return all_results


async def _eval_single_ragas(item: QueryItem, param_group, metrics: dict, kb_id: int = 1) -> dict:
    """单条 RAGAS 评估：检索 + 生成答案 + 评估"""
    from app.clients.milvus_utils import (
        get_milvus_client, create_hybrid_search_requests,
        hybrid_search, fetch_chunks_by_chunk_ids
    )
    from app.lm.embedding_utils import generate_embeddings
    from app.lm.lm_utils import get_llm_client
    from langchain_core.messages import HumanMessage, SystemMessage

    relevant = set(item.relevant_chunk_ids)
    client = get_milvus_client()
    if not client:
        return {"question": item.question, "error": "Milvus 不可用"}

    # 1. 向量检索（带 kb_id + item_name 过滤）
    embeddings = generate_embeddings([item.question])
    dense_vec = embeddings.get("dense", [[]])[0]
    sparse_vec = embeddings.get("sparse", [[]])[0]

    if not dense_vec:
        return {"question": item.question, "error": "向量生成为空"}

    expr = f"kb_id == {kb_id}"
    if item.item_name:
        safe_name = item.item_name.replace('"', '\\"')
        expr += f' and item_name == "{safe_name}"'

    reqs = create_hybrid_search_requests(
        dense_vec, sparse_vec,
        expr=expr,
        limit=param_group.top_k * 3,
    )
    weights = (param_group.dense_weight, param_group.sparse_weight)

    res = hybrid_search(
        client=client, collection_name=param_group.collection_name,
        reqs=reqs, ranker_weights=weights, limit=param_group.top_k,
        output_fields=["chunk_id"],
    )
    chunk_ids = _extract_chunk_ids(res)[:param_group.top_k]

    # 2. 获取 chunk 文本
    contexts = []
    if chunk_ids:
        chunks = fetch_chunks_by_chunk_ids(
            client, param_group.collection_name, chunk_ids,
            output_fields=["chunk_id", "content"]
        )
        for c in chunks:
            ctx = c.get("content", "")
            if ctx:
                contexts.append(str(ctx))

    # 3. LLM 生成答案
    generated_answer = ""
    if contexts:
        try:
            llm_client = get_llm_client()
            context_text = "\n\n---\n\n".join(contexts[:5])
            messages = [
                SystemMessage(content=(
                    "你是一个知识库问答助手。请根据提供的参考文档回答用户问题。"
                    "如果文档中没有相关信息，请如实说明。回答要简洁准确。"
                )),
                HumanMessage(content=(
                    f"参考文档：\n{context_text}\n\n用户问题：{item.question}\n\n请回答："
                )),
            ]
            response = llm_client.invoke(messages)
            generated_answer = response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            logger.warning(f"生成答案失败: {e}")

    # 4. RAGAS 评估
    scores = {}
    ctx_list = contexts if contexts else ["无相关内容"]
    ans = generated_answer or "无法回答"

    if generated_answer:
        for name, metric in metrics.items():
            try:
                result = await metric.ascore(
                    user_input=item.question,
                    response=ans,
                    retrieved_contexts=ctx_list,
                )
                scores[name] = round(float(result.value), 4)
            except Exception as e:
                logger.warning(f"RAGAS 指标 [{name}] 计算失败: {e}")
                scores[name] = 0.0
    else:
        for name in metrics:
            scores[name] = 0.0

    return {
        "question": item.question,
        "expected_answer": item.expected_answer,
        "generated_answer": ans[:300],
        "retrieved_contexts": contexts,
        "hit": any(rid in relevant for rid in chunk_ids),
        **scores,
    }


def _extract_chunk_ids(search_result) -> list[str]:
    """从 Milvus 搜索结果提取 chunk_id（主键存在 hit['id']）"""
    ids = []
    if search_result and len(search_result) > 0:
        for hit in search_result[0]:
            cid = hit.get("id")
            if cid is not None:
                ids.append(str(cid))
    return ids
