"""
Query 改写节点：将口语化/指代不明的用户查询改写为规范的产品手册查询语句。
位于意图确认之后、多路搜索之前，提升检索召回质量。
"""
import sys
from app.utils.task_utils import add_running_task, add_done_task
from app.core.load_prompt import load_prompt
from app.core.logger import logger
from app.lm.lm_utils import get_llm_client


def node_query_rewrite(state):
    """
    将 rewritten_query 进一步优化为适合检索的规范化查询语句。
    仅在有 item_names 确认的情况下执行改写；无 item_names 时原样透传。
    """
    logger.info("---query_rewrite (查询改写) 开始处理---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    rewritten_query = state.get("rewritten_query") or state.get("original_query", "")
    item_names = state.get("item_names", [])

    # 无商品名时不改写，原样透传
    if not item_names or not rewritten_query:
        logger.info("query_rewrite: 无商品名或无查询内容，跳过改写")
        add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
        return {}

    try:
        llm = get_llm_client()
        item_names_str = "、".join(item_names)
        prompt = load_prompt("query_rewrite", item_names=item_names_str, query=rewritten_query)
        response = llm.invoke(prompt)
        improved_query = (response.content or "").strip()

        if improved_query and len(improved_query) > 3:
            logger.info(f"query_rewrite: 改写完成 [{rewritten_query[:30]}...] → [{improved_query[:60]}...]")
            state["rewritten_query"] = improved_query
        else:
            logger.warning("query_rewrite: LLM 返回内容过短，保留原查询")
    except Exception as e:
        logger.error(f"query_rewrite: 改写失败，保留原查询: {e}")

    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    return {"rewritten_query": state["rewritten_query"]}
