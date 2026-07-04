"""检索评估 API 路由 - 数据集管理、评估任务、结果查询"""
import csv
import io
import json
import threading
from datetime import datetime

from fastapi import APIRouter, HTTPException, UploadFile, File

from app.eval.models import EvalDataset, QueryItem, EvalTask
from app.eval.db import datasets_col, items_col, tasks_col
from app.eval.engine import run_eval_task
from app.eval.ragas_engine import run_ragas_eval
from app.core.logger import logger

router = APIRouter(prefix="/eval", tags=["检索评估"])


# ═══════════════ 数据集管理 ═══════════════

@router.get("/datasets")
def list_datasets(page: int = 1, size: int = 20, keyword: str = ""):
    """数据集列表，支持分页和关键词搜索"""
    query = {}
    if keyword:
        query["name"] = {"$regex": keyword, "$options": "i"}
    total = datasets_col().count_documents(query)
    cursor = datasets_col().find(query).skip((page - 1) * size).limit(size)
    items = []
    for doc in cursor:
        doc.pop("_id", None)
        items.append(doc)
    return {"total": total, "page": page, "size": size, "items": items}


@router.post("/datasets")
def create_dataset(dataset: EvalDataset):
    """创建空数据集"""
    datasets_col().insert_one(dataset.model_dump())
    return {"dataset_id": dataset.dataset_id}


@router.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: str):
    """数据集详情，含测试条目列表"""
    doc = datasets_col().find_one({"dataset_id": dataset_id})
    if not doc:
        raise HTTPException(404, "数据集不存在")
    doc.pop("_id", None)
    cursor = items_col().find({"dataset_id": dataset_id})
    items = []
    for item in cursor:
        item.pop("_id", None)
        items.append(item)
    doc["items"] = items
    return doc


@router.delete("/datasets/{dataset_id}")
def delete_dataset(dataset_id: str):
    """删除数据集及所有关联条目"""
    items_col().delete_many({"dataset_id": dataset_id})
    datasets_col().delete_one({"dataset_id": dataset_id})
    return {"ok": True}


@router.post("/datasets/{dataset_id}/items")
def add_item(dataset_id: str, item: QueryItem):
    """添加单条测试条目"""
    item.dataset_id = dataset_id
    items_col().insert_one(item.model_dump())
    datasets_col().update_one(
        {"dataset_id": dataset_id},
        {"$inc": {"query_count": 1}, "$set": {"updated_at": datetime.now().isoformat()}}
    )
    return {"item_id": item.item_id}


@router.post("/datasets/{dataset_id}/import")
async def import_items(dataset_id: str, file: UploadFile = File(...)):
    """上传 JSON/CSV 文件批量导入条目"""
    content = await file.read()
    text = content.decode("utf-8")
    items = []

    if file.filename and file.filename.endswith(".json"):
        data = json.loads(text)
        if isinstance(data, list):
            for row in data:
                items.append(QueryItem(
                    question=row["question"],
                    item_name=row.get("item_name", ""),
                    relevant_chunk_ids=row.get("relevant_chunk_ids", []),
                    expected_answer=row.get("expected_answer", ""),
                ))
    elif file.filename and file.filename.endswith(".csv"):
        reader = csv.DictReader(io.StringIO(text))
        for row in reader:
            items.append(QueryItem(
                question=row["question"],
                item_name=row.get("item_name", ""),
                relevant_chunk_ids=[c.strip() for c in row.get("relevant_chunk_ids", "").split(",") if c.strip()],
                expected_answer=row.get("expected_answer", ""),
            ))
    else:
        raise HTTPException(400, "仅支持 .json 和 .csv 格式")

    for item in items:
        item.dataset_id = dataset_id
        items_col().insert_one(item.model_dump())

    count = len(items)
    datasets_col().update_one(
        {"dataset_id": dataset_id},
        {"$inc": {"query_count": count}, "$set": {"updated_at": datetime.now().isoformat()}}
    )
    logger.info(f"导入 {count} 条测试条目到数据集 [{dataset_id}]")
    return {"imported": count}


@router.post("/datasets/{dataset_id}/items/batch")
def batch_add_items(dataset_id: str, items: list[QueryItem]):
    """批量添加测试条目（从文档生成数据集时使用）"""
    if not items:
        raise HTTPException(400, "items 不能为空")
    count = 0
    for item in items:
        item.dataset_id = dataset_id
        items_col().insert_one(item.model_dump())
        count += 1
    datasets_col().update_one(
        {"dataset_id": dataset_id},
        {"$inc": {"query_count": count}, "$set": {"updated_at": datetime.now().isoformat()}}
    )
    logger.info(f"批量添加 {count} 条条目到数据集 [{dataset_id}]")
    return {"added": count}


@router.put("/datasets/{dataset_id}/items/{item_id}")
def update_item(dataset_id: str, item_id: str, item: QueryItem):
    """更新单条测试条目"""
    result = items_col().update_one(
        {"dataset_id": dataset_id, "item_id": item_id},
        {"$set": {
            "question": item.question,
            "item_name": item.item_name,
            "relevant_chunk_ids": item.relevant_chunk_ids,
            "expected_answer": item.expected_answer,
        }}
    )
    if result.matched_count == 0:
        raise HTTPException(404, "条目不存在")
    return {"ok": True}


@router.delete("/datasets/{dataset_id}/items/{item_id}")
def delete_item(dataset_id: str, item_id: str):
    """删除单条测试条目"""
    result = items_col().delete_one({"dataset_id": dataset_id, "item_id": item_id})
    datasets_col().update_one(
        {"dataset_id": dataset_id},
        {"$inc": {"query_count": -1}, "$set": {"updated_at": datetime.now().isoformat()}}
    )
    return {"ok": True, "deleted": result.deleted_count}


# ═══════════════ 文档切片查询（用于生成数据集） ═══════════════

@router.get("/chunks")
def list_chunks_by_file(kb_id: int = 1, file_title: str = ""):
    """按文件标题查询 Milvus 中的切片列表，用于数据集生成预览"""
    if not file_title:
        raise HTTPException(400, "file_title 不能为空")
    from app.clients.milvus_utils import get_milvus_client, query_chunks_by_file_title
    import os
    client = get_milvus_client()
    if not client:
        raise HTTPException(500, "Milvus 连接失败")
    collection = os.getenv("CHUNKS_COLLECTION", "kaas_chunks")
    chunks = query_chunks_by_file_title(client, collection, file_title, kb_id=kb_id)
    # 只返回必要字段，减少传输量
    items = []
    for c in chunks:
        items.append({
            "chunk_id": c.get("id") or c.get("chunk_id"),
            "content": c.get("content", "")[:500],  # 截断预览
            "title": c.get("title", ""),
            "item_name": c.get("item_name", ""),
            "part": c.get("part", 0),
        })
    return {"file_title": file_title, "total": len(items), "chunks": items}


# ═══════════════ 评估任务 ═══════════════

@router.post("/tasks")
def create_task(task: EvalTask):
    """创建并启动评估任务（后台线程执行）"""
    dataset = datasets_col().find_one({"dataset_id": task.dataset_id})
    if not dataset:
        raise HTTPException(404, "数据集不存在")

    cursor = items_col().find({"dataset_id": task.dataset_id})
    dataset_items = []
    for doc in cursor:
        doc.pop("_id", None)
        dataset_items.append(QueryItem(**doc))

    if not dataset_items:
        raise HTTPException(400, "数据集中没有测试条目")

    tasks_col().insert_one(task.model_dump())

    # 根据 eval_mode 选择引擎
    if task.eval_mode == "ragas":
        engine_func = run_ragas_eval
    else:
        engine_func = run_eval_task

    thread = threading.Thread(target=engine_func, args=(task, dataset_items), daemon=True)
    thread.start()

    return {
        "task_id": task.task_id,
        "status": "running",
        "eval_mode": task.eval_mode,
        "query_count": len(dataset_items),
    }


@router.get("/tasks")
def list_tasks(page: int = 1, size: int = 10):
    """评估任务列表"""
    total = tasks_col().count_documents({})
    cursor = tasks_col().find({}).sort("created_at", -1).skip((page - 1) * size).limit(size)
    items = []
    for doc in cursor:
        doc.pop("_id", None)
        items.append(doc)
    return {"total": total, "page": page, "size": size, "items": items}


@router.get("/tasks/{task_id}")
def get_task(task_id: str):
    """任务状态和结果摘要"""
    doc = tasks_col().find_one({"task_id": task_id})
    if not doc:
        raise HTTPException(404, "任务不存在")
    doc.pop("_id", None)
    return doc


@router.post("/tasks/{task_id}/cancel")
def cancel_task(task_id: str):
    """取消运行中的任务"""
    tasks_col().update_one(
        {"task_id": task_id, "status": "running"},
        {"$set": {"status": "cancelled", "finished_at": datetime.now().isoformat()}}
    )
    return {"ok": True}


# ═══════════════ 结果查询 ═══════════════

@router.get("/tasks/{task_id}/results")
def get_task_results(task_id: str):
    """单任务完整结果"""
    doc = tasks_col().find_one({"task_id": task_id})
    if not doc:
        raise HTTPException(404, "任务不存在")
    return {"results": doc.get("results", {})}


@router.get("/tasks/{task_id}/compare")
def compare_results(task_id: str):
    """多组参数横向对比"""
    doc = tasks_col().find_one({"task_id": task_id})
    if not doc:
        raise HTTPException(404, "任务不存在")
    results = doc.get("results", {})
    compare_rows = []
    for group_key, result in results.items():
        compare_rows.append({
            "group": group_key,
            "params": result.get("params", {}),
            "overall": result.get("overall_metrics", {}),
            "stage": result.get("stage_metrics", {}),
        })
    return {"compare": compare_rows}
