import hashlib
import os
import shutil
import uuid
from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException, Form
from fastapi.responses import FileResponse

from app.clients.minio_utils import get_minio_client
from app.clients.file_record_dao import insert_record, find_by_hash, get_all_records
from app.clients.kb_dao import get_kb_config, increment_doc_count
from app.utils.path_util import PROJECT_ROOT
from app.utils.task_utils import (
    add_running_task,
    add_done_task,
    get_done_task_list,
    get_running_task_list,
    get_task_status,
)
from app.api.services.import_service import run_graph_task
from app.core.logger import logger

router = APIRouter(tags=["知识库导入"])


def _compute_md5(file_path: str) -> str:
    """分块计算文件 MD5，避免大文件内存溢出"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


@router.get("/")
async def get_index_page():
    """返回统一前端页面"""
    html_abs_path = PROJECT_ROOT / "app/page/index.html"
    logger.info(f"前端页面访问，文件绝对路径：{html_abs_path}")

    if not os.path.exists(html_abs_path):
        logger.error(f"前端页面文件不存在，路径：{html_abs_path}")
        raise HTTPException(status_code=404, detail="index.html page not found")

    return FileResponse(path=html_abs_path, media_type="text/html")


@router.post("/upload", summary="文件上传接口")
async def upload_files(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    kb_id: int = Form(1),
):
    """文件上传核心接口：接收文件 → 本地保存 → MinIO上传 → 启动后台任务"""
    date_based_root_dir = os.path.join(PROJECT_ROOT / "output", datetime.now().strftime("%Y%m%d"))
    task_ids = []

    for file in files:
        task_id = str(uuid.uuid4())
        task_ids.append(task_id)
        logger.info(f"[{task_id}] ========== 上传流程开始 ==========")
        logger.info(f"[{task_id}] [步骤1/5] 接收文件: {file.filename}")

        add_running_task(task_id, "upload_file")

        task_local_dir = os.path.join(date_based_root_dir, task_id)
        os.makedirs(task_local_dir, exist_ok=True)
        local_file_abs_path = os.path.join(task_local_dir, file.filename)

        with open(local_file_abs_path, "wb") as file_buffer:
            shutil.copyfileobj(file.file, file_buffer)
        logger.info(f"[{task_id}] [步骤2/5] 本地保存完成: {local_file_abs_path}")

        # 去重检查：计算 MD5 并查询是否已存在
        file_hash = _compute_md5(local_file_abs_path)
        logger.info(f"[{task_id}] [去重检查] 文件MD5: {file_hash}")

        existing = find_by_hash(file_hash, kb_id=kb_id)
        if existing:
            logger.info(f"[{task_id}] [去重检查] 发现重复文件: existing_id={existing['id']}, "
                        f"existing_task_id={existing['task_id']}")
            # 清理本地临时文件
            shutil.rmtree(task_local_dir, ignore_errors=True)
            return {
                "code": 409,
                "message": "文件已存在",
                "existing": {
                    "id": existing["id"],
                    "file_name": existing["file_name"],
                    "file_size": existing["file_size"],
                    "created_at": str(existing["created_at"]),
                    "task_id": existing["task_id"],
                }
            }

        # MinIO 上传
        minio_pdf_base_dir = os.getenv("MINIO_PDF_DIR", "pdf_files")
        minio_object_name = f"{minio_pdf_base_dir}/{datetime.now().strftime('%Y%m%d')}/{file.filename}"
        minio_bucket_name = os.getenv("MINIO_BUCKET_NAME", "kb-import-bucket")
        file_type = "pdf" if file.filename.lower().endswith(".pdf") else "md"
        file_size = os.path.getsize(local_file_abs_path)
        logger.info(f"[{task_id}] [步骤3/5] 写入MySQL记录: type={file_type}, size={file_size}, minio_obj={minio_object_name}")

        # 先写入 MySQL 文件记录，确保列表始终有记录
        record_id = insert_record(
            file_name=file.filename,
            file_size=file_size,
            file_type=file_type,
            file_hash=file_hash,
            minio_bucket=minio_bucket_name,
            minio_object=minio_object_name,
            task_id=task_id,
            kb_id=kb_id,
        )
        increment_doc_count(kb_id)
        logger.info(f"[{task_id}] [步骤3/5] MySQL写入成功, record_id={record_id}")

        # MinIO 上传（失败不阻塞流程，仅记录日志）
        logger.info(f"[{task_id}] [步骤4/5] 上传MinIO: bucket={minio_bucket_name}")
        try:
            minio_client = get_minio_client()
            if minio_client is None:
                logger.warning(f"[{task_id}] [步骤4/5] MinIO客户端不可用，文件未上传至对象存储")
            else:
                minio_client.fput_object(
                    bucket_name=minio_bucket_name,
                    object_name=minio_object_name,
                    file_path=local_file_abs_path,
                    content_type=file.content_type
                )
                logger.info(f"[{task_id}] [步骤4/5] MinIO上传成功")
        except Exception as e:
            logger.warning(f"[{task_id}] [步骤4/5] MinIO上传失败: {e}", exc_info=True)

        add_done_task(task_id, "upload_file")
        background_tasks.add_task(run_graph_task, task_id, task_local_dir, local_file_abs_path, kb_id)
        logger.info(f"[{task_id}] [步骤5/5] 后台任务已提交, local_dir={task_local_dir}, kb_id={kb_id}")
        logger.info(f"[{task_id}] ========== 上传流程结束 ==========")

    return {
        "code": 200,
        "message": f"Files uploaded successfully, total: {len(files)}",
        "task_ids": task_ids
    }


@router.post("/upload/overwrite", summary="覆盖上传接口")
async def overwrite_upload(
    background_tasks: BackgroundTasks,
    old_task_id: str = Form(...),
    file: UploadFile = File(...),
    kb_id: int = Form(1),
):
    """覆盖上传：删除旧文件的 MinIO 对象和 Milvus 向量，重新导入新文件"""
    from app.clients.file_record_dao import get_record_by_task_id, delete_record
    from app.clients.milvus_utils import get_milvus_client

    task_id = str(uuid.uuid4())
    logger.info(f"[{task_id}] ========== 覆盖上传开始，旧task_id={old_task_id}, kb_id={kb_id} ==========")

    old_record = get_record_by_task_id(old_task_id)
    if not old_record:
        raise HTTPException(status_code=404, detail="旧文件记录不存在")

    # 1. 删除 MinIO 旧文件
    try:
        minio_client = get_minio_client()
        if minio_client:
            minio_client.remove_object(old_record["minio_bucket"], old_record["minio_object"])
            logger.info(f"[{task_id}] MinIO 旧文件已删除: {old_record['minio_object']}")
    except Exception as e:
        logger.warning(f"[{task_id}] MinIO 旧文件删除失败: {e}")

    # 2. 删除 Milvus 向量
    try:
        milvus_client = get_milvus_client()
        if milvus_client:
            collection_name = os.getenv("CHUNKS_COLLECTION", "kb_chunks")
            milvus_client.delete(
                collection_name=collection_name,
                filter=f'task_id == "{old_task_id}"'
            )
            logger.info(f"[{task_id}] Milvus 向量已删除: task_id={old_task_id}")
    except Exception as e:
        logger.warning(f"[{task_id}] Milvus 向量删除失败: {e}")

    # 3. 删除 MySQL 旧记录
    delete_record(old_record["id"])

    # 4. 正常上传流程
    add_running_task(task_id, "upload_file")

    date_based_root_dir = os.path.join(PROJECT_ROOT / "output", datetime.now().strftime("%Y%m%d"))
    task_local_dir = os.path.join(date_based_root_dir, task_id)
    os.makedirs(task_local_dir, exist_ok=True)
    local_file_abs_path = os.path.join(task_local_dir, file.filename)

    with open(local_file_abs_path, "wb") as file_buffer:
        shutil.copyfileobj(file.file, file_buffer)
    logger.info(f"[{task_id}] 本地保存完成: {local_file_abs_path}")

    file_hash = _compute_md5(local_file_abs_path)
    minio_pdf_base_dir = os.getenv("MINIO_PDF_DIR", "pdf_files")
    minio_object_name = f"{minio_pdf_base_dir}/{datetime.now().strftime('%Y%m%d')}/{file.filename}"
    minio_bucket_name = os.getenv("MINIO_BUCKET_NAME", "kb-import-bucket")
    file_type = "pdf" if file.filename.lower().endswith(".pdf") else "md"
    file_size = os.path.getsize(local_file_abs_path)

    record_id = insert_record(
        file_name=file.filename,
        file_size=file_size,
        file_type=file_type,
        file_hash=file_hash,
        minio_bucket=minio_bucket_name,
        minio_object=minio_object_name,
        task_id=task_id,
        kb_id=kb_id,
    )

    try:
        minio_client = get_minio_client()
        if minio_client:
            minio_client.fput_object(
                bucket_name=minio_bucket_name,
                object_name=minio_object_name,
                file_path=local_file_abs_path,
                content_type=file.content_type
            )
    except Exception as e:
        logger.warning(f"[{task_id}] MinIO上传失败: {e}")

    add_done_task(task_id, "upload_file")
    background_tasks.add_task(run_graph_task, task_id, task_local_dir, local_file_abs_path, kb_id)

    return {
        "code": 200,
        "message": "覆盖上传成功",
        "task_id": task_id
    }


@router.get("/status/{task_id}", summary="任务状态查询")
async def get_task_progress(task_id: str):
    """根据TaskID查询处理进度"""
    task_status_info: Dict[str, Any] = {
        "code": 200,
        "task_id": task_id,
        "status": get_task_status(task_id),
        "done_list": get_done_task_list(task_id),
        "running_list": get_running_task_list(task_id)
    }
    logger.info(f"[{task_id}] 任务状态查询，当前状态：{task_status_info['status']}")
    return task_status_info


@router.get("/files", summary="文件列表")
async def list_files(kb_id: int = None):
    records = get_all_records(kb_id=kb_id)
    return {"code": 200, "data": records}


@router.delete("/files/{file_id}", summary="删除文件")
async def delete_file(file_id: int):
    from app.clients.file_record_dao import get_record_by_id, delete_record
    from app.clients.milvus_utils import get_milvus_client

    record = get_record_by_id(file_id)
    if not record:
        raise HTTPException(status_code=404, detail="文件记录不存在")

    # 1. 删除 MinIO 文件
    try:
        minio_client = get_minio_client()
        if minio_client:
            minio_client.remove_object(record["minio_bucket"], record["minio_object"])
            logger.info(f"MinIO 文件已删除: {record['minio_object']}")
    except Exception as e:
        logger.warning(f"MinIO 文件删除失败: {e}")

    # 2. 删除 Milvus 向量（按 task_id 过滤）
    if record.get("task_id"):
        try:
            milvus_client = get_milvus_client()
            if milvus_client:
                collection_name = os.getenv("CHUNKS_COLLECTION", "kb_chunks")
                milvus_client.delete(
                    collection_name=collection_name,
                    filter=f'task_id == "{record["task_id"]}"'
                )
                logger.info(f"Milvus 向量已删除: task_id={record['task_id']}")
        except Exception as e:
            logger.warning(f"Milvus 向量删除失败: {e}")

    # 3. 删除 MySQL 记录
    delete_record(file_id)

    # 4. 清理本地临时文件
    date_based_root_dir = os.path.join(PROJECT_ROOT / "output", datetime.now().strftime("%Y%m%d"))
    task_dir = os.path.join(date_based_root_dir, record.get("task_id", ""))
    if os.path.exists(task_dir):
        shutil.rmtree(task_dir, ignore_errors=True)
        logger.info(f"本地临时文件已清理: {task_dir}")

    return {"code": 200, "message": "删除成功"}
