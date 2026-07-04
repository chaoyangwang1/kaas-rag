"""检索评估模块 - MongoDB 操作"""
import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = MongoClient(
            os.getenv("MONGO_URL"),
            serverSelectionTimeoutMS=10000,
            connectTimeoutMS=10000,
        )
    return _client


def _db():
    return _get_client()[os.getenv("MONGO_DB_NAME")]


def datasets_col():
    return _db()["eval_datasets"]


def items_col():
    return _db()["eval_query_items"]


def tasks_col():
    return _db()["eval_tasks"]
