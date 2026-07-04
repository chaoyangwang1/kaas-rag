"""
knowledge_bases 表 DAO（数据访问对象）
提供知识库的增删改查功能，模块导入时自动建表
"""
import json
from app.clients.mysql_utils import db
from app.core.logger import logger

TABLE_NAME = "knowledge_bases"

# 默认知识库配置
DEFAULT_KB_CONFIG = {
    "import": {
        "chunk_max_length": 2000,
        "chunk_min_length": 500,
    },
    "entity_match": {
        "dense_weight": 0.8,
        "sparse_weight": 0.2,
    },
    "embedding_search": {
        "dense_weight": 0.8,
        "sparse_weight": 0.2,
        "req_limit": 10,
        "top_k": 5,
    },
    "hyde_search": {
        "dense_weight": 0.8,
        "sparse_weight": 0.2,
        "req_limit": 10,
        "top_k": 5,
    },
    "rrf": {
        "k": 60,
        "max_results": 10,
        "embedding_weight": 1.0,
        "hyde_weight": 1.0,
    },
    "rerank": {
        "max_topk": 10,
        "min_topk": 1,
        "gap_ratio": 0.25,
        "gap_abs": 0.5,
    },
    "features": {
        "enable_hyde": True,
        "enable_web_search": True,
        "enable_kg_search": True,
        "enable_query_rewrite": True,
    },
    "generate": {
        "max_context_chars": 12000,
        "temperature": 0.1,
    },
}


def _log(action, detail=""):
    logger.info(f"[kb_dao] {action}" + (f" | {detail}" if detail else ""))


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS knowledge_bases (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    name            VARCHAR(200)    NOT NULL COMMENT '知识库名称',
    description     TEXT            COMMENT '知识库描述',
    config_json     TEXT            COMMENT '可调参数JSON',
    doc_count       INT             DEFAULT 0 COMMENT '已导入文档数',
    created_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE INDEX idx_kb_name (name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='知识库表';
"""


def ensure_table():
    db.execute(CREATE_TABLE_SQL)
    _log("建表", "knowledge_bases 已就绪")


def _ensure_default_kb():
    """确保默认知识库存在（id=1），用于向后兼容"""
    existing = get_kb_by_id(1)
    if not existing:
        db.insert(
            f"INSERT INTO {TABLE_NAME} (id, name, description, config_json) VALUES (%s, %s, %s, %s)",
            (1, "默认知识库", "系统默认知识库，升级前已有数据归入此库",
             json.dumps(DEFAULT_KB_CONFIG, ensure_ascii=False))
        )
        _log("初始化", "默认知识库(id=1)已创建")


def get_kb_by_id(kb_id: int):
    return db.fetchone(f"SELECT * FROM {TABLE_NAME} WHERE id = %s", (kb_id,))


def get_all_kbs():
    records = db.fetchall(
        f"SELECT id, name, description, doc_count, created_at, updated_at "
        f"FROM {TABLE_NAME} ORDER BY id ASC"
    )
    _log("LIST", f"共 {len(records)} 个知识库")
    return records


def get_kb_config(kb_id: int) -> dict:
    """获取知识库的可调参数配置（JSON → dict），每次查询直读MySQL保证实时生效"""
    record = get_kb_by_id(kb_id)
    if not record:
        _log("GET_CONFIG", f"kb_id={kb_id} 不存在，返回默认配置")
        return dict(DEFAULT_KB_CONFIG)
    try:
        raw = record.get("config_json", "") or ""
        config = json.loads(raw) if raw.strip() else {}
        # 空配置或无有效配置项时回退到默认
        if not isinstance(config, dict) or len(config) == 0:
            _log("GET_CONFIG", f"kb_id={kb_id} 配置为空，返回默认配置")
            return dict(DEFAULT_KB_CONFIG)
        _log("GET_CONFIG", f"kb_id={kb_id} 配置已加载，包含 {len(config)} 个顶层配置项")
        logger.debug(f"[kb_dao] kb_id={kb_id} 当前配置: {json.dumps(config, ensure_ascii=False)[:200]}")
        return config
    except (json.JSONDecodeError, TypeError) as e:
        _log("GET_CONFIG", f"kb_id={kb_id} 配置解析失败({e})，返回默认配置")
        return dict(DEFAULT_KB_CONFIG)


def create_kb(name: str, description: str = "", config: dict = None) -> int:
    config_json = json.dumps(config or DEFAULT_KB_CONFIG, ensure_ascii=False)
    kb_id = db.insert(
        f"INSERT INTO {TABLE_NAME} (name, description, config_json) VALUES (%s, %s, %s)",
        (name, description, config_json)
    )
    _log("CREATE", f"kb_id={kb_id}, name={name}")
    return kb_id


def update_kb(kb_id: int, name: str = None, description: str = None, config: dict = None):
    fields = []
    params = []
    if name is not None:
        fields.append("name = %s")
        params.append(name)
    if description is not None:
        fields.append("description = %s")
        params.append(description)
    if config is not None:
        fields.append("config_json = %s")
        params.append(json.dumps(config, ensure_ascii=False))
    if not fields:
        return
    params.append(kb_id)
    db.execute(f"UPDATE {TABLE_NAME} SET {', '.join(fields)} WHERE id = %s", params)
    _log("UPDATE", f"kb_id={kb_id}")


def delete_kb(kb_id: int):
    db.execute(f"DELETE FROM {TABLE_NAME} WHERE id = %s", (kb_id,))
    _log("DELETE", f"kb_id={kb_id}")


def increment_doc_count(kb_id: int):
    db.execute(f"UPDATE {TABLE_NAME} SET doc_count = doc_count + 1 WHERE id = %s", (kb_id,))


def decrement_doc_count(kb_id: int):
    db.execute(
        f"UPDATE {TABLE_NAME} SET doc_count = GREATEST(doc_count - 1, 0) WHERE id = %s",
        (kb_id,)
    )


# 模块导入时自动建表 + 初始化默认知识库
ensure_table()
_ensure_default_kb()
