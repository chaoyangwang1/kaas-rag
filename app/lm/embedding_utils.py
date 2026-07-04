import hashlib
from functools import lru_cache
import requests
import numpy as np
from scipy.sparse import csr_matrix
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from app.core.logger import logger
from app.conf.embedding_config import embedding_config

# 模型单例对象，避免重复初始化
_bge_m3_ef = None
# 本地 BGE-M3 模型单例
_bge_m3_local = None

# Embedding 结果缓存（进程内 LRU）
# key: MD5(text), value: {"dense": [...], "sparse": [...]}
_embedding_cache: dict = {}
_EMBEDDING_CACHE_MAX_SIZE = 1000
_EMBEDDING_CACHE_HIT = 0
_EMBEDDING_CACHE_MISS = 0


class LocalEmbeddingClient:
    """本地 BGE-M3 模型客户端，直接生成稠密 + 稀疏向量"""

    def __init__(self, model_path: str, use_fp16: bool = True, device: str = "cpu"):
        from FlagEmbedding import BGEM3FlagModel
        logger.info(f"正在加载本地 BGE-M3 模型: {model_path} (use_fp16={use_fp16}, device={device})")
        self._model = BGEM3FlagModel(model_path, use_fp16=use_fp16, devices=device)
        logger.success(f"本地 BGE-M3 模型加载成功: {model_path}")

    def encode_documents(self, texts: list[str]) -> dict:
        """
        使用本地 BGE-M3 生成稠密 + 稀疏向量
        返回格式: {"dense": [ndarray, ...], "sparse": csr_matrix}
        """
        output = self._model.encode(texts, return_dense=True, return_sparse=True)
        dense = output.get("dense_vecs", [])
        lexical_weights = output.get("lexical_weights", [])

        # 构建稀疏矩阵
        sparse_indices_list = []
        sparse_values_list = []
        indptr = [0]
        for lw in lexical_weights:
            if lw and isinstance(lw, dict):
                indices = list(lw.keys())
                values = list(lw.values())
            else:
                indices = []
                values = []
            sparse_indices_list.extend(indices)
            sparse_values_list.extend(values)
            indptr.append(indptr[-1] + len(indices))

        total_elements = len(sparse_indices_list)
        sparse = csr_matrix(
            (sparse_values_list if total_elements else [0.0],
             sparse_indices_list if total_elements else [0],
             indptr),
            shape=(len(texts), 1024)
        )

        return {"dense": [np.array(d, dtype=np.float32) for d in dense], "sparse": sparse}


class RemoteEmbeddingClient:
    """HTTP 包装器，对调用方伪装为 BGEM3EmbeddingFunction"""

    def __init__(self, base_url: str, api_key: str, model_name: str):
        self._url = f"{base_url.rstrip('/')}/embeddings"
        self._headers = {}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"
        self._model = model_name

        # 创建带重试机制的 Session，应对远程服务偶发断连（RemoteDisconnected/5xx）
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,  # 重试间隔：1s → 2s → 4s（指数退避）
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"],
        )
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=1,  # 限制连接池大小，避免 Keep-Alive 连接复用导致的 RemoteDisconnected
            pool_maxsize=1,
        )
        self._session = requests.Session()
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)
        # 强制每次请求新建连接（Connection: close），防止 Keep-Alive 连接被服务端提前关闭
        self._session.headers["Connection"] = "close"

        logger.info(f"RemoteEmbeddingClient 初始化: url={self._url}, model={self._model}, retry=3")

    def encode_documents(self, texts: list[str]) -> dict:
        """
        调用远程 /v1/embeddings
        返回格式与 BGEM3EmbeddingFunction.encode_documents() 一致:
        {"dense": [ndarray, ...], "sparse": csr_matrix}
        """
        # 合并请求头：显式加 Connection: close 防止 Keep-Alive 导致的 RemoteDisconnected
        req_headers = {**self._headers, "Connection": "close"}
        resp = self._session.post(
            self._url,
            json={"model": self._model, "input": texts, "encoding_format": "float"},
            headers=req_headers,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()

        # 稠密向量：标准 OpenAI 格式
        dense = [np.array(d["embedding"], dtype=np.float32) for d in data["data"]]

        # 稀疏向量：优先从远程响应提取，远程没有则本地生成
        sparse = self._build_sparse_matrix(data["data"], len(texts))
        # 检查稀疏矩阵是否为空（远程未返回稀疏数据），是则用本地 BGE-M3 生成
        if sparse.nnz == 0:
            logger.info("远程服务未返回稀疏向量，使用本地 BGE-M3 生成稀疏向量")
            sparse = _generate_sparse_locally(texts)

        return {"dense": dense, "sparse": sparse}

    def _build_sparse_matrix(self, data: list, num_texts: int):
        """从响应中构建 CSR 稀疏矩阵，与原 BGEM3EmbeddingFunction 输出一致"""
        sparse_indices_list = []
        sparse_values_list = []
        indptr = [0]

        for i, item in enumerate(data):
            if "sparse" in item and item["sparse"]:
                sparse = item["sparse"]
                if isinstance(sparse, dict):
                    indices = list(sparse.keys())
                    values = list(sparse.values())
                else:
                    indices = []
                    values = []
            else:
                indices = []
                values = []
            sparse_indices_list.extend(indices)
            sparse_values_list.extend(values)
            indptr.append(indptr[-1] + len(indices))

        total_elements = len(sparse_indices_list)
        return csr_matrix(
            (sparse_values_list if total_elements else [0.0],
             sparse_indices_list if total_elements else [0],
             indptr),
            shape=(num_texts, 1024)
        )


def get_bge_m3_ef():
    """获取 embedding 客户端单例（根据配置决定使用本地模型或远程服务）"""
    global _bge_m3_ef
    if _bge_m3_ef is not None:
        logger.debug("Embedding 客户端单例已存在，直接返回")
        return _bge_m3_ef

    if embedding_config.use_local_model:
        # 使用本地 BGE-M3 模型
        model_path = embedding_config.local_model_path or "BAAI/bge-m3"
        logger.info(f"配置为本地模式，加载模型: {model_path}")
        try:
            _bge_m3_ef = LocalEmbeddingClient(
                model_path,
                use_fp16=embedding_config.use_fp16,
                device=embedding_config.device,
            )
            return _bge_m3_ef
        except Exception as e:
            logger.error(f"本地 BGE-M3 模型加载失败：{e}", exc_info=True)
            raise
    else:
        # 使用远程 Embedding 服务
        logger.info("配置为远程模式，初始化 Embedding 客户端")
        try:
            _bge_m3_ef = RemoteEmbeddingClient(
                base_url=embedding_config.base_url,
                api_key=embedding_config.api_key,
                model_name=embedding_config.model_name
            )
            logger.success("远程 Embedding 客户端初始化成功")
            return _bge_m3_ef
        except Exception as e:
            logger.error(f"远程 Embedding 客户端初始化失败：{str(e)}", exc_info=True)
            raise


def get_bge_m3_local():
    """获取本地 BGE-M3 模型单例（远程模式下用于稀疏向量生成回退）"""
    global _bge_m3_local
    if _bge_m3_local is not None:
        return _bge_m3_local

    model_path = embedding_config.local_model_path or "BAAI/bge-m3"
    try:
        from FlagEmbedding import BGEM3FlagModel
        _bge_m3_local = BGEM3FlagModel(
            model_path,
            use_fp16=embedding_config.use_fp16,
            devices=embedding_config.device,
        )
        logger.success(f"本地 BGE-M3 模型加载成功（路径: {model_path}, use_fp16={embedding_config.use_fp16}, device={embedding_config.device}）")
        return _bge_m3_local
    except Exception as e:
        logger.warning(f"本地 BGE-M3 模型加载失败，稀疏向量将为空: {e}")
        return None


def _generate_sparse_locally(texts: list) -> csr_matrix:
    """使用本地 BGE-M3 模型生成稀疏向量（CSR 格式）"""
    model = get_bge_m3_local()
    if model is None:
        logger.warning("本地 BGE-M3 不可用，返回空稀疏矩阵")
        return csr_matrix((len(texts), 1024))

    output = model.encode(texts, return_dense=False, return_sparse=True)
    lexical_weights = output.get("lexical_weights", [])

    sparse_indices_list = []
    sparse_values_list = []
    indptr = [0]
    for lw in lexical_weights:
        if lw and isinstance(lw, dict):
            indices = list(lw.keys())
            values = list(lw.values())
        else:
            indices = []
            values = []
        sparse_indices_list.extend(indices)
        sparse_values_list.extend(values)
        indptr.append(indptr[-1] + len(indices))

    total_elements = len(sparse_indices_list)
    if total_elements == 0:
        return csr_matrix((len(texts), 1024))

    return csr_matrix(
        (sparse_values_list, sparse_indices_list, indptr),
        shape=(len(texts), 1024)
    )


MAX_TEXT_LENGTH = 500  # 远程嵌入服务请求体限制极严，超长文本截断


def _cache_key(texts: list) -> str:
    """为文本列表生成缓存键（MD5 哈希）"""
    combined = "|".join(texts)
    return hashlib.md5(combined.encode("utf-8")).hexdigest()


def generate_embeddings(texts, use_cache: bool = True):
    """
    为文本列表生成稠密+稀疏混合向量嵌入
    超长文本自动截断，避免远程嵌入服务413错误
    支持 LRU 进程内缓存，热点 query 直接命中。
    :param texts: 要生成嵌入的文本列表，单文本也需封装为列表
    :param use_cache: 是否启用缓存（默认开启）
    :return: 字典格式的向量结果，key为dense/sparse
    """
    global _EMBEDDING_CACHE_HIT, _EMBEDDING_CACHE_MISS

    if not isinstance(texts, list) or len(texts) == 0:
        logger.warning("生成向量入参不合法，texts必须为非空列表")
        raise ValueError("参数texts必须是包含文本的非空列表")

    # 缓存查找
    if use_cache:
        ck = _cache_key(texts)
        if ck in _embedding_cache:
            _EMBEDDING_CACHE_HIT += 1
            logger.debug(f"Embedding缓存命中 (总命中={_EMBEDDING_CACHE_HIT}, "
                         f"命中率={_EMBEDDING_CACHE_HIT/(_EMBEDDING_CACHE_HIT+_EMBEDDING_CACHE_MISS):.1%})")
            return _embedding_cache[ck]
        _EMBEDDING_CACHE_MISS += 1

    # 远程模式下超长文本截断，防止413 Payload Too Large；本地模式不截断
    if embedding_config.use_local_model:
        truncated_texts = texts
    else:
        truncated_texts = []
        for t in texts:
            if len(t) > MAX_TEXT_LENGTH:
                truncated_texts.append(t[:MAX_TEXT_LENGTH])
                logger.warning(f"文本过长({len(t)}字符)，已截断至{MAX_TEXT_LENGTH}字符")
            else:
                truncated_texts.append(t)

    logger.info(f"开始为{len(texts)}条文本生成混合向量嵌入")
    try:
        model = get_bge_m3_ef()
        embeddings = model.encode_documents(truncated_texts)
        logger.debug(f"模型编码完成，开始解析稀疏向量格式，共{len(texts)}条")

        processed_sparse = []
        for i in range(len(texts)):
            sparse = embeddings["sparse"]
            sparse_indices = sparse.indices[
                sparse.indptr[i]:sparse.indptr[i + 1]
            ].tolist()
            sparse_data = sparse.data[
                sparse.indptr[i]:sparse.indptr[i + 1]
            ].tolist()
            sparse_dict = {int(k): float(v) for k, v in zip(sparse_indices, sparse_data)}
            processed_sparse.append(sparse_dict)

        result = {
            "dense": [emb.tolist() for emb in embeddings["dense"]],
            "sparse": processed_sparse
        }
        logger.success(f"{len(texts)}条文本向量生成完成")

        # 写入缓存（LRU 淘汰）
        if use_cache:
            ck = _cache_key(texts)
            if len(_embedding_cache) >= _EMBEDDING_CACHE_MAX_SIZE:
                # 淘汰最早的一个 key（近似 LRU）
                oldest_key = next(iter(_embedding_cache))
                _embedding_cache.pop(oldest_key, None)
                logger.debug(f"Embedding缓存已满，淘汰旧条目")
            _embedding_cache[ck] = result

        return result

    except Exception as e:
        logger.error(f"文本向量生成失败：{str(e)}", exc_info=True)
        raise
