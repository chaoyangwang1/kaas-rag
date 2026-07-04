from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


class QueryRequest(BaseModel):
    """查询请求数据结构"""
    query: str = Field(..., description="查询内容")
    session_id: str = Field(None, description="会话ID")
    is_stream: bool = Field(False, description="是否流式返回")
    kb_id: int = Field(1, description="知识库ID，默认1")


class OverwriteRequest(BaseModel):
    """覆盖上传请求"""
    old_task_id: str = Field(..., description="要覆盖的旧文件任务ID")


# ===== 知识库管理模型 =====

class KBCreateRequest(BaseModel):
    """创建知识库请求"""
    name: str = Field(..., description="知识库名称")
    description: str = Field("", description="知识库描述")
    config: Optional[Dict[str, Any]] = Field(None, description="可调参数配置JSON")


class KBUpdateRequest(BaseModel):
    """更新知识库请求"""
    name: Optional[str] = Field(None, description="知识库名称")
    description: Optional[str] = Field(None, description="知识库描述")
    config: Optional[Dict[str, Any]] = Field(None, description="可调参数配置JSON")


class KBResponse(BaseModel):
    """知识库响应"""
    id: int
    name: str
    description: str
    doc_count: int
    config_json: Optional[Dict[str, Any]] = None
    created_at: str = ""
    updated_at: str = ""
