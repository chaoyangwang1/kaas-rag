"""检索评估模块 - 数据模型"""
import uuid
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


def _uid():
    return str(uuid.uuid4())


def _now():
    return datetime.now().isoformat()


class QueryItem(BaseModel):
    """单条测试条目：问题 + 标注的相关文档ID"""
    item_id: str = Field(default_factory=_uid)
    dataset_id: str = ""
    question: str
    item_name: str = ""  # 关联的商品名称，用于检索过滤
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    expected_answer: str = ""


class EvalDataset(BaseModel):
    """数据集：包含多条测试问题"""
    dataset_id: str = Field(default_factory=_uid)
    name: str
    description: str = ""
    query_count: int = 0
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)


class ParamGroup(BaseModel):
    """一组检索参数 - 对应 KB 配置中的一个检索段"""
    name: str = ""  # 参数组名称标识（如 entity_match / embedding_search / hyde_search）
    dense_weight: float = 0.8
    sparse_weight: float = 0.2
    req_limit: int = 10
    top_k: int = 5
    collection_name: str = "kaas_chunks"


class EvalTask(BaseModel):
    """评估任务"""
    task_id: str = Field(default_factory=_uid)
    dataset_id: str
    kb_id: int = 1  # 关联的知识库ID
    status: str = "pending"
    eval_mode: str = "retrieval"  # retrieval | ragas
    param_groups: list[ParamGroup] = Field(default_factory=list)
    results: dict = Field(default_factory=dict)
    created_at: str = Field(default_factory=_now)
    finished_at: Optional[str] = None


class PerQueryDetail(BaseModel):
    """单个问题的评估详情"""
    question: str
    relevant_ids: list[str] = Field(default_factory=list)
    retrieved_ids: list[str] = Field(default_factory=list)
    rank: Optional[int] = None
    hit: bool = False


class StageMetrics(BaseModel):
    """分阶段召回指标"""
    embedding_recall: float = 0.0
    hyde_recall: float = 0.0
    rrf_recall: float = 0.0
    rerank_recall: float = 0.0


class EvalResult(BaseModel):
    """一组参数的完整评估结果"""
    group_index: int
    params: dict
    overall_metrics: dict = Field(default_factory=dict)
    stage_metrics: StageMetrics = Field(default_factory=StageMetrics)
    per_query_details: list[dict] = Field(default_factory=list)
