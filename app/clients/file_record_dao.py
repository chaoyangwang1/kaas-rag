"""
file_records 表 DAO（数据访问对象）
提供文件上传记录的增删改查功能，模块导入时自动建表
"""
from app.clients.mysql_utils import db
from app.core.logger import logger

TABLE_NAME = "file_records"

def _log(action, detail=""):
    """统一日志前缀"""
    logger.info(f"[file_record_dao] {action}" + (f" | {detail}" if detail else ""))

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS file_records (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    kb_id           INT             NOT NULL DEFAULT 1 COMMENT '所属知识库ID',
    file_name       VARCHAR(500)    NOT NULL COMMENT '原始文件名',
    file_size       BIGINT          NOT NULL COMMENT '文件大小(字节)',
    file_type       VARCHAR(10)     NOT NULL COMMENT '文件类型: pdf/md',
    file_hash       VARCHAR(64)     NOT NULL DEFAULT '' COMMENT '文件MD5哈希',
    minio_bucket    VARCHAR(100)    NOT NULL COMMENT 'MinIO存储桶名',
    minio_object    VARCHAR(1000)   NOT NULL COMMENT 'MinIO对象路径',
    embedding_status VARCHAR(20)    NOT NULL DEFAULT 'uploaded' COMMENT 'uploaded/processing/completed/failed',
    task_id         VARCHAR(36)              COMMENT '关联任务ID(UUID)',
    error_message   TEXT                    COMMENT '失败时的错误信息',
    created_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_kb_id (kb_id),
    INDEX idx_task_id (task_id),
    INDEX idx_status (embedding_status),
    INDEX idx_file_hash (file_hash),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='文件上传记录表';
"""


def ensure_table():
    db.execute(CREATE_TABLE_SQL)
    _log("建表", "file_records 已就绪")


def insert_record(file_name, file_size, file_type, file_hash, minio_bucket, minio_object, task_id, kb_id=1):
    record_id = db.insert(
        f"INSERT INTO {TABLE_NAME} (kb_id, file_name, file_size, file_type, file_hash, minio_bucket, minio_object, embedding_status, task_id) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, 'uploaded', %s)",
        (kb_id, file_name, file_size, file_type, file_hash, minio_bucket, minio_object, task_id)
    )
    _log("INSERT", f"record_id={record_id}, file={file_name}, type={file_type}, hash={file_hash}, kb_id={kb_id}, task_id={task_id}")
    return record_id


def get_all_records(kb_id=None):
    if kb_id is not None:
        records = db.fetchall(
            f"SELECT id, kb_id, file_name, file_size, file_type, embedding_status, task_id, error_message, created_at, updated_at "
            f"FROM {TABLE_NAME} WHERE kb_id = %s ORDER BY created_at DESC",
            (kb_id,)
        )
    else:
        records = db.fetchall(
            f"SELECT id, kb_id, file_name, file_size, file_type, embedding_status, task_id, error_message, created_at, updated_at "
            f"FROM {TABLE_NAME} ORDER BY created_at DESC"
        )
    _log("SELECT_ALL", f"共 {len(records)} 条记录, kb_id={kb_id}")
    return records


def get_record_by_id(record_id):
    record = db.fetchone(f"SELECT * FROM {TABLE_NAME} WHERE id = %s", (record_id,))
    _log("SELECT_BY_ID", f"record_id={record_id}, found={record is not None}")
    return record


def get_record_by_task_id(task_id):
    record = db.fetchone(f"SELECT * FROM {TABLE_NAME} WHERE task_id = %s", (task_id,))
    _log("SELECT_BY_TASK", f"task_id={task_id}, found={record is not None}")
    return record


def update_status(record_id, status, error_message=None):
    db.execute(
        f"UPDATE {TABLE_NAME} SET embedding_status = %s, error_message = %s WHERE id = %s",
        (status, error_message, record_id)
    )
    _log("UPDATE_STATUS", f"record_id={record_id}, status={status}, err={error_message}")


def update_status_by_task_id(task_id, status, error_message=None):
    db.execute(
        f"UPDATE {TABLE_NAME} SET embedding_status = %s, error_message = %s WHERE task_id = %s",
        (status, error_message, task_id)
    )
    _log("UPDATE_STATUS", f"task_id={task_id}, status={status}, err={error_message}")


def delete_record(record_id):
    db.execute(f"DELETE FROM {TABLE_NAME} WHERE id = %s", (record_id,))
    _log("DELETE", f"record_id={record_id}")


def find_by_hash(file_hash, kb_id=1):
    """根据 MD5 哈希在指定知识库内查询未失败的记录（用于去重检测）"""
    record = db.fetchone(
        f"SELECT id, file_name, file_size, file_type, embedding_status, task_id, created_at "
        f"FROM {TABLE_NAME} WHERE file_hash = %s AND kb_id = %s AND embedding_status != 'failed' "
        f"ORDER BY created_at DESC LIMIT 1",
        (file_hash, kb_id)
    )
    _log("FIND_BY_HASH", f"hash={file_hash[:16]}..., kb_id={kb_id}, found={record is not None}")
    return record


def get_doc_count_by_kb(kb_id, status=None):
    """统计指定知识库下指定状态的文档数量（用于删除保护判断）"""
    if status:
        record = db.fetchone(
            f"SELECT COUNT(*) as cnt FROM {TABLE_NAME} WHERE kb_id = %s AND embedding_status = %s",
            (kb_id, status)
        )
    else:
        record = db.fetchone(
            f"SELECT COUNT(*) as cnt FROM {TABLE_NAME} WHERE kb_id = %s",
            (kb_id,)
        )
    count = record["cnt"] if record else 0
    _log("COUNT_BY_KB", f"kb_id={kb_id}, status={status}, count={count}")
    return count
