from app.utils.task_utils import update_task_status, reset_task_progress, TASK_STATUS_COMPLETED, TASK_STATUS_FAILED
from app.utils.sse_utils import SSEEvent, push_to_session
from app.query_process.agent.main_graph import query_app
from app.clients.kb_dao import get_kb_config
from app.core.logger import logger


def run_query_graph(session_id: str, user_query: str, is_stream: bool = True, kb_id: int = 1):
    """查询图后台任务"""
    logger.info(f"开始查询流程: session={session_id}, kb_id={kb_id}, query={user_query[:50]}...")

    # 重置上一轮查询的进度状态，避免多轮对话进度叠加
    reset_task_progress(session_id)

    # 加载知识库配置
    kb_config = get_kb_config(kb_id)

    default_state = {
        "original_query": user_query,
        "session_id": session_id,
        "is_stream": is_stream,
        "kb_id": kb_id,
        "kb_config": kb_config,
    }
    try:
        query_app.invoke(default_state)
        update_task_status(session_id, TASK_STATUS_COMPLETED, is_stream)
    except Exception as e:
        logger.error(f"查询流程异常: {e}", exc_info=True)
        update_task_status(session_id, TASK_STATUS_FAILED, is_stream)
        if is_stream:
            push_to_session(session_id, SSEEvent.ERROR, {"error": str(e)})
