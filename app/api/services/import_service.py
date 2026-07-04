import os

from app.utils.task_utils import (
    add_done_task,
    update_task_status,
)
from app.clients.file_record_dao import update_status_by_task_id
from app.clients.kb_dao import get_kb_config
from app.import_process.agent.state import get_default_state
from app.import_process.agent.main_graph import kb_import_app
from app.core.logger import logger


def run_graph_task(task_id: str, local_dir: str, local_file_path: str, kb_id: int = 1):
    """LangGraph 全流程执行后台任务"""
    try:
        logger.info(f"[{task_id}] ========== 图表执行开始 (kb_id={kb_id}) ==========")
        logger.info(f"[{task_id}] [准备阶段] 状态更新为 processing")
        update_task_status(task_id, "processing")
        update_status_by_task_id(task_id, "processing")

        # 加载知识库配置
        kb_config = get_kb_config(kb_id)

        # 优先使用本地已有文件，本地缺失时从 MinIO 下载
        logger.info(f"[{task_id}] [准备阶段] 检查文件: {local_file_path}")
        if not os.path.exists(local_file_path):
            logger.warning(f"[{task_id}] [准备阶段] 本地文件不存在，尝试从MinIO下载")
            from app.clients.file_record_dao import get_record_by_task_id
            from app.clients.minio_utils import get_minio_client

            record = get_record_by_task_id(task_id)
            if record:
                logger.info(f"[{task_id}] [准备阶段] MySQL记录存在: bucket={record['minio_bucket']}, object={record['minio_object']}")
                try:
                    minio_client = get_minio_client()
                    if minio_client:
                        download_path = os.path.join(local_dir, os.path.basename(record["minio_object"]))
                        logger.info(f"[{task_id}] [准备阶段] 开始MinIO下载: -> {download_path}")
                        minio_client.fget_object(record["minio_bucket"], record["minio_object"], download_path)
                        local_file_path = download_path
                        logger.info(f"[{task_id}] [准备阶段] MinIO下载成功: {download_path}")
                except Exception as e:
                    logger.warning(f"[{task_id}] [准备阶段] MinIO下载失败: {e}")
            else:
                logger.warning(f"[{task_id}] [准备阶段] MySQL记录不存在")
            if not os.path.exists(local_file_path):
                raise FileNotFoundError(f"[{task_id}] 文件不存在，本地和MinIO均无法获取: {local_file_path}")
        else:
            logger.info(f"[{task_id}] [准备阶段] 本地文件存在，直接使用: {local_file_path}")

        init_state = get_default_state()
        init_state["task_id"] = task_id
        init_state["local_dir"] = local_dir
        init_state["local_file_path"] = local_file_path
        init_state["kb_id"] = kb_id
        init_state["kb_config"] = kb_config
        logger.info(f"[{task_id}] [准备阶段] init_state 构造完成, kb_id={kb_id}, 开始执行图表")

        logger.info(f"[{task_id}] [图表执行] 开始 stream...")
        for event in kb_import_app.stream(init_state):
            for node_name, node_result in event.items():
                logger.info(f"[{task_id}] [图表执行] 节点完成: {node_name}")
                add_done_task(task_id, node_name)
        logger.info(f"[{task_id}] [图表执行] stream 结束")

        update_task_status(task_id, "completed")
        update_status_by_task_id(task_id, "completed")
        logger.info(f"[{task_id}] ========== 图表执行完成 (completed) ==========")

    except Exception as e:
        update_task_status(task_id, "failed")
        update_status_by_task_id(task_id, "failed", str(e))
        logger.error(f"[{task_id}] ========== 图表执行失败 (failed) ==========")
        logger.error(f"[{task_id}] 异常类型: {type(e).__name__}, 异常信息: {str(e)}", exc_info=True)
