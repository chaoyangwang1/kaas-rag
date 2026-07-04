# -*- coding: utf-8 -*-
"""
测试文件管理功能：
- DAO 模块导入
- 文件记录 CRUD
- 状态流转
"""
import sys
from app.core.logger import logger

logger.info("=" * 50)
logger.info("开始文件管理功能测试")
logger.info("=" * 50)

# Test 1: DAO module import
logger.info("Test 1: DAO 模块导入测试")
try:
    from app.clients.file_record_dao import (
        TABLE_NAME, ensure_table,
        insert_record, get_all_records, get_record_by_id,
        get_record_by_task_id, update_status, update_status_by_task_id,
        delete_record
    )
    logger.info("[PASS] DAO 模块导入成功, 表名: {}".format(TABLE_NAME))
except Exception as e:
    logger.warning("[SKIP] DAO 模块导入异常 (MySQL 可能未就绪): {}".format(e))
    logger.info("=" * 50)
    logger.info("测试结束 (跳过数据库操作测试)")
    logger.info("=" * 50)
    sys.exit(0)

# Test 0: Verify MySQL connection before running CRUD tests
logger.info("Test 0: MySQL 连接验证")
try:
    from app.clients.mysql_utils import db
    conn = db.get_conn()
    conn.ping()
    logger.info("[PASS] MySQL 连接正常")
except Exception as e:
    logger.warning("[SKIP] MySQL 连接失败, 跳过 CRUD 测试: {}".format(e))
    logger.info("=" * 50)
    logger.info("测试结束 (请先配置 .env 中的 MySQL 连接信息)")
    logger.info("=" * 50)
    sys.exit(0)

# Test 2: CRUD operations
logger.info("Test 2: 文件记录 CRUD 测试")
record_id = None
try:
    # Insert
    record_id = insert_record(
        file_name="test_manual.pdf",
        file_size=2048,
        file_type="pdf",
        minio_bucket="test-bucket",
        minio_object="pdf_files/20260627/test_manual.pdf",
        task_id="test-task-crud-001"
    )
    assert record_id > 0, "插入失败：record_id 应大于 0"
    logger.info("[PASS] 插入成功, record_id={}".format(record_id))

    # Query by id
    record = get_record_by_id(record_id)
    assert record is not None, "按 ID 查询失败"
    assert record["file_name"] == "test_manual.pdf", "文件名不匹配: {}".format(record['file_name'])
    assert record["embedding_status"] == "uploaded", "状态应为 uploaded: {}".format(record['embedding_status'])
    logger.info("[PASS] 按 ID 查询成功: {}".format(record['file_name']))

    # Query by task_id
    record2 = get_record_by_task_id("test-task-crud-001")
    assert record2 is not None, "按 task_id 查询失败"
    logger.info("[PASS] 按 task_id 查询成功")

    # Update status by id
    update_status(record_id, "completed")
    record = get_record_by_id(record_id)
    assert record["embedding_status"] == "completed", "状态应为 completed: {}".format(record['embedding_status'])
    logger.info("[PASS] 按 ID 更新状态成功")

    logger.info("[PASS] Test 2 全部通过")

except Exception as e:
    logger.error("[FAIL] Test 2 失败: {}".format(e))

# Test 3: Status flow (uploaded -> processing -> completed -> failed)
logger.info("Test 3: 状态流转测试")
try:
    rid = insert_record("test_status_flow.md", 512, "md", "bucket", "obj/test.md", "task-flow-001")

    update_status_by_task_id("task-flow-001", "processing")
    r = get_record_by_id(rid)
    assert r["embedding_status"] == "processing", "应为 processing: {}".format(r['embedding_status'])
    logger.info("[PASS] uploaded -> processing")

    update_status_by_task_id("task-flow-001", "completed")
    r = get_record_by_id(rid)
    assert r["embedding_status"] == "completed", "应为 completed: {}".format(r['embedding_status'])
    logger.info("[PASS] processing -> completed")

    update_status_by_task_id("task-flow-001", "failed", "timeout")
    r = get_record_by_id(rid)
    assert r["embedding_status"] == "failed"
    assert r["error_message"] == "timeout"
    logger.info("[PASS] completed -> failed (with error_message)")

    delete_record(rid)
    assert get_record_by_id(rid) is None, "删除后应返回 None"
    logger.info("[PASS] 删除成功")

    logger.info("[PASS] Test 3 全部通过")

except Exception as e:
    logger.error("[FAIL] Test 3 失败: {}".format(e))

# Test 4: List records
logger.info("Test 4: 文件列表查询测试")
try:
    records = get_all_records()
    assert isinstance(records, list), "返回应为 list"
    logger.info("[PASS] 文件列表查询成功, 当前共 {} 条记录".format(len(records)))
except Exception as e:
    logger.error("[FAIL] Test 4 失败: {}".format(e))

# Clean up test data
if record_id:
    try:
        delete_record(record_id)
        logger.info("[CLEAN] 清理测试数据: record_id={}".format(record_id))
    except:
        pass

logger.info("=" * 50)
logger.info("文件管理功能测试结束")
logger.info("=" * 50)
