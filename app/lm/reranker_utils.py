import requests
from app.conf.reranker_config import reranker_config
from app.core.logger import logger

_reranker_model = None


class RemoteRerankerClient:
    """HTTP 包装器，对调用方伪装为 FlagReranker"""

    def __init__(self, base_url: str, api_key: str, model_name: str):
        self._url = f"{base_url.rstrip('/')}/rerank"
        self._headers = {}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"
        self._model = model_name
        logger.info(f"RemoteRerankerClient 初始化: url={self._url}, model={self._model}")

    def compute_score(self, sentence_pairs: list[tuple[str, str]]) -> list[float]:
        """
        调用远程 /v1/rerank
        sentence_pairs: [(query, passage), ...]
        返回: [float, ...]
        """
        query = sentence_pairs[0][0]
        documents = [pair[1] for pair in sentence_pairs]

        resp = requests.post(
            self._url,
            json={"model": self._model, "query": query, "documents": documents},
            headers=self._headers,
            timeout=60
        )
        resp.raise_for_status()
        data = resp.json()

        results = sorted(data["results"], key=lambda r: r["index"])
        return [r["relevance_score"] for r in results]


def get_reranker_model():
    """获取 reranker 客户端单例（远程服务）"""
    global _reranker_model
    if _reranker_model is None:
        logger.info("开始初始化远程 Reranker 客户端")
        _reranker_model = RemoteRerankerClient(
            base_url=reranker_config.base_url,
            api_key=reranker_config.api_key,
            model_name=reranker_config.model_name
        )
        logger.success("远程 Reranker 客户端初始化成功")
    return _reranker_model
