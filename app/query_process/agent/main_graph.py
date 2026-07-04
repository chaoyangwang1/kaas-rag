from langgraph.graph import StateGraph, END
from langgraph.constants import Send
from app.query_process.agent.state import QueryGraphState
# 导入所有节点函数
from app.query_process.agent.nodes.node_item_name_confirm import node_item_name_confirm
from app.query_process.agent.nodes.node_query_rewrite import node_query_rewrite
from app.query_process.agent.nodes.node_query_kg import node_query_kg
from app.query_process.agent.nodes.node_answer_output import node_answer_output
from app.query_process.agent.nodes.node_rerank import node_rerank
from app.query_process.agent.nodes.node_rrf import node_rrf
from app.query_process.agent.nodes.node_search_embedding import node_search_embedding
from app.query_process.agent.nodes.node_search_embedding_hyde import node_search_embedding_hyde
from app.query_process.agent.nodes.node_web_search_mcp import node_web_search_mcp

# 初始化状态图
builder = StateGraph(QueryGraphState)

# 注册所有节点
builder.add_node("node_item_name_confirm", node_item_name_confirm)  # 确认商品
builder.add_node("node_query_rewrite", node_query_rewrite)          # 查询改写（新增）
builder.add_node("node_multi_search", lambda x: x)                  # 虚拟节点：多路搜索分叉点
builder.add_node("node_search_embedding", node_search_embedding)    # 向量搜索
builder.add_node("node_search_embedding_hyde", node_search_embedding_hyde)
builder.add_node("node_query_kg", node_query_kg)
builder.add_node("node_web_search_mcp", node_web_search_mcp)
builder.add_node("node_join", lambda x: {})  # 虚拟节点：多路搜索合并点
builder.add_node("node_rrf", node_rrf)        # 排序
builder.add_node("node_rerank", node_rerank)  # 重排
builder.add_node("node_answer_output", node_answer_output)  # 生成

# 设置起点
builder.set_entry_point("node_item_name_confirm")


def route_after_item_confirm(state: QueryGraphState):
    """
    意图确认后的条件路由：
    - 已有 answer（反问/拒答场景）→ 直接输出
    - 启用 Query 改写 → 进入查询改写
    - 未启用 → 直接进入多路搜索
    """
    if state.get("answer"):
        return "node_answer_output"
    kb_config = state.get("kb_config", {})
    features = kb_config.get("features", {}) or {}
    if features.get("enable_query_rewrite", True):
        return "node_query_rewrite"
    return "node_multi_search"


def fanout_to_searches(state: QueryGraphState):
    """
    使用 LangGraph Send API 并行分叉到搜索节点。
    根据知识库的功能开关决定启用哪些搜索路径。
    """
    kb_config = state.get("kb_config", {})
    features = kb_config.get("features", {}) or {}

    searches = [Send("node_search_embedding", state)]  # 向量搜索始终开启

    if features.get("enable_hyde", True):
        searches.append(Send("node_search_embedding_hyde", state))
    if features.get("enable_web_search", True):
        searches.append(Send("node_web_search_mcp", state))
    if features.get("enable_kg_search", True):
        searches.append(Send("node_query_kg", state))

    return searches


# 1. 意图确认 → 查询改写 → 多路搜索 / 直接输出
builder.add_conditional_edges("node_item_name_confirm", route_after_item_confirm)
builder.add_edge("node_query_rewrite", "node_multi_search")

# 2. 使用 Send API 实现四路搜索的真正并行执行（替代原来 4 条独立的 add_edge）
builder.add_conditional_edges(
    "node_multi_search",
    fanout_to_searches,
    path_map=["node_search_embedding", "node_search_embedding_hyde",
              "node_web_search_mcp", "node_query_kg"]
)

# 3. 四路搜索 → 结果合并
builder.add_edge("node_search_embedding", "node_join")
builder.add_edge("node_search_embedding_hyde", "node_join")
builder.add_edge("node_web_search_mcp", "node_join")
builder.add_edge("node_query_kg", "node_join")

# 4. 合并 → 排序 → 重排 → 生成 → 结束
builder.add_edge("node_join", "node_rrf")
builder.add_edge("node_rrf", "node_rerank")
builder.add_edge("node_rerank", "node_answer_output")
builder.add_edge("node_answer_output", END)

# 编译生成可执行的 Runnable 应用
query_app = builder.compile()
