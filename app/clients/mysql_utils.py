import pymysql
import threading
from pymysql.cursors import DictCursor
from app.conf.mysql_config import mysql_config
from app.core.logger import logger

# MySQL连接管理类（线程本地存储，保证多线程并发安全）
class MySQLPool:
    """
    MySQL 连接池封装类
    使用线程本地存储，每个线程独立连接，避免并发读写冲突
    """

    def __init__(self, config):
        self.config = config
        self._local = threading.local()

    def _create_conn(self):
        return pymysql.connect(
            host=self.config.host,
            port=self.config.port,
            user=self.config.user,
            password=self.config.password,
            database=self.config.database,
            charset=self.config.charset,
            cursorclass=DictCursor,
            autocommit=True,
        )

    def get_conn(self):
        """
        获取当前线程的MySQL连接（线程隔离，自动重连）
        :return: pymysql.Connection 连接对象
        """
        try:
            conn = getattr(self._local, 'connection', None)
            if conn is None or not conn.open:
                self._local.connection = self._create_conn()
                logger.info("MySQL 连接成功（线程本地）")
            else:
                conn.ping(reconnect=True)
            return self._local.connection
        except Exception as e:
            logger.error(f"MySQL 连接失败: {e}", exc_info=True)
            raise

    def execute(self, sql, params=None):
        """
        执行SQL语句（INSERT/UPDATE/DELETE/DDL），自动关闭游标
        """
        conn = self.get_conn()
        with conn.cursor() as cursor:
            cursor.execute(sql, params or ())
        conn.commit()

    def fetchone(self, sql, params=None):
        """
        查询单条记录
        """
        conn = self.get_conn()
        with conn.cursor() as cursor:
            cursor.execute(sql, params or ())
            return cursor.fetchone()

    def fetchall(self, sql, params=None):
        """
        查询多条记录
        """
        conn = self.get_conn()
        with conn.cursor() as cursor:
            cursor.execute(sql, params or ())
            return cursor.fetchall()

    def insert(self, sql, params=None):
        """
        插入记录并返回自增ID
        """
        conn = self.get_conn()
        with conn.cursor() as cursor:
            cursor.execute(sql, params or ())
            last_id = cursor.lastrowid
        conn.commit()
        return last_id

# 全局MySQL连接池实例，供全项目调用（单例模式）
db = MySQLPool(mysql_config)
