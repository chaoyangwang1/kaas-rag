"""知识库管理 API 路由"""
from fastapi import APIRouter, HTTPException

from app.api.models import KBCreateRequest, KBUpdateRequest
from app.clients.kb_dao import (
    create_kb, get_all_kbs, get_kb_by_id, get_kb_config,
    update_kb, delete_kb,
)
from app.clients.file_record_dao import get_doc_count_by_kb
from app.core.logger import logger

router = APIRouter(prefix="/api/kb", tags=["知识库管理"])


@router.get("", summary="获取所有知识库")
async def list_kbs():
    records = get_all_kbs()
    return {"code": 200, "data": records}


@router.post("", summary="创建知识库")
async def create(req: KBCreateRequest):
    kb_id = create_kb(
        name=req.name,
        description=req.description,
        config=req.config
    )
    logger.info(f"知识库创建成功: id={kb_id}, name={req.name}")
    return {"code": 201, "message": "知识库创建成功", "kb_id": kb_id}


@router.get("/{kb_id}", summary="获取知识库详情")
async def get_detail(kb_id: int):
    record = get_kb_by_id(kb_id)
    if not record:
        raise HTTPException(status_code=404, detail="知识库不存在")
    config = get_kb_config(kb_id)
    record["config_json"] = config
    return {"code": 200, "data": record}


@router.get("/{kb_id}/config", summary="获取知识库配置")
async def get_config(kb_id: int):
    config = get_kb_config(kb_id)
    return {"code": 200, "data": config}


@router.put("/{kb_id}", summary="更新知识库")
async def update(kb_id: int, req: KBUpdateRequest):
    existing = get_kb_by_id(kb_id)
    if not existing:
        raise HTTPException(status_code=404, detail="知识库不存在")
    update_kb(
        kb_id=kb_id,
        name=req.name,
        description=req.description,
        config=req.config
    )
    logger.info(f"知识库更新成功: id={kb_id}")
    return {"code": 200, "message": "知识库更新成功"}


@router.delete("/{kb_id}", summary="删除知识库")
async def delete(kb_id: int):
    existing = get_kb_by_id(kb_id)
    if not existing:
        raise HTTPException(status_code=404, detail="知识库不存在")

    # 删除保护：有已学习（completed）文档时拒绝
    completed_count = get_doc_count_by_kb(kb_id, status="completed")
    if completed_count > 0:
        raise HTTPException(
            status_code=409,
            detail=f"该知识库下有 {completed_count} 个文档处于已学习（完成）状态，无法删除。"
                   f"请先逐个删除知识库内的文档后再删除知识库。"
        )

    # 检查是否有处理中的文档
    in_progress = get_doc_count_by_kb(kb_id, status="processing") + get_doc_count_by_kb(kb_id, status="uploaded")
    if in_progress > 0:
        raise HTTPException(
            status_code=409,
            detail=f"该知识库下有 {in_progress} 个文档正在处理中，无法删除。请等待处理完成或先删除这些文档。"
        )

    delete_kb(kb_id)
    logger.info(f"知识库删除成功: id={kb_id}, name={existing.get('name')}")
    return {"code": 200, "message": "知识库已删除"}
