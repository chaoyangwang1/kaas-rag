import json
import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.api.models import QueryRequest
from app.api.services.query_service import run_query_graph
from app.utils.task_utils import *
from app.utils.sse_utils import create_sse_queue, sse_generator
from app.clients.mongo_history_utils import *

router = APIRouter(tags=["智能查询"])


@router.get("/health")
async def health():
    """健康检查"""
    return {"ok": True}


@router.post("/query")
async def query(background_tasks: BackgroundTasks, request: QueryRequest):
    """用户提问接口，支持同步和流式两种模式"""
    user_query = request.query
    session_id = request.session_id if request.session_id else str(uuid.uuid4())
    is_stream = request.is_stream
    kb_id = request.kb_id

    if is_stream:
        create_sse_queue(session_id)

    update_task_status(session_id, TASK_STATUS_PROCESSING, is_stream)

    if is_stream:
        background_tasks.add_task(run_query_graph, session_id, user_query, is_stream, kb_id)
        return {
            "message": "结果正在处理中...",
            "session_id": session_id
        }
    else:
        run_query_graph(session_id, user_query, is_stream, kb_id)
        answer = get_task_result(session_id, "answer", "")
        entity_options_raw = get_task_result(session_id, "entity_options", "")
        entity_options = json.loads(entity_options_raw) if entity_options_raw else []
        return {
            "message": "处理完成！",
            "session_id": session_id,
            "answer": answer,
            "entity_options": entity_options,
            "done_list": []
        }


@router.get("/stream/{session_id}")
async def stream(session_id: str, request: Request):
    """SSE 流式返回结果"""
    return StreamingResponse(
        sse_generator(session_id, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.get("/history/{session_id}")
async def history(session_id: str, limit: int = 50):
    """查询当前会话历史记录"""
    try:
        records = get_recent_messages(session_id, limit=limit)
        items = []
        for r in records:
            items.append({
                "_id": str(r.get("_id")) if r.get("_id") is not None else "",
                "session_id": r.get("session_id", ""),
                "role": r.get("role", ""),
                "text": r.get("text", ""),
                "rewritten_query": r.get("rewritten_query", ""),
                "item_names": r.get("item_names", []),
                "ts": r.get("ts")
            })
        return {"session_id": session_id, "items": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"history error: {e}")


@router.delete("/history/{session_id}")
async def clear_chat_history(session_id: str):
    """清除会话历史"""
    count = clear_history(session_id)
    return {"message": "History cleared", "deleted_count": count}
