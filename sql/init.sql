-- ============================================
-- KAAS 知识库系统 - 数据库初始化脚本
-- 执行方式: mysql -u root -p < sql/init.sql
-- ============================================

-- 1. 创建数据库（如不存在）
CREATE DATABASE IF NOT EXISTS kaas_rag
    DEFAULT CHARACTER SET utf8mb4
    DEFAULT COLLATE utf8mb4_unicode_ci;

USE kaas_rag;

-- 2. 创建文件上传记录表
CREATE TABLE IF NOT EXISTS file_records (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    file_name       VARCHAR(500)    NOT NULL COMMENT '原始文件名',
    file_size       BIGINT          NOT NULL COMMENT '文件大小(字节)',
    file_type       VARCHAR(10)     NOT NULL COMMENT '文件类型: pdf / md',
    minio_bucket    VARCHAR(100)    NOT NULL COMMENT 'MinIO 存储桶名',
    minio_object    VARCHAR(1000)   NOT NULL COMMENT 'MinIO 对象路径',
    embedding_status VARCHAR(20)    NOT NULL DEFAULT 'uploaded' COMMENT 'uploaded / processing / completed / failed',
    task_id         VARCHAR(36)              COMMENT '关联任务 ID (UUID)',
    error_message   TEXT                    COMMENT '失败时的错误信息',
    created_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    INDEX idx_task_id (task_id),
    INDEX idx_status (embedding_status),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='文件上传记录表';
