# 导入核心依赖：数据类、环境变量读取
from dataclasses import dataclass
import os
from dotenv import load_dotenv

# 提前加载.env配置文件
load_dotenv()

@dataclass
class RerankerConfig:
    base_url: str      # 远程服务地址，如 http://10.0.0.5:8001/v1
    api_key: str       # API密钥，不需要认证则留空
    model_name: str    # 模型名称，如 bge-reranker-large

# 实例化配置对象
reranker_config = RerankerConfig(
    base_url=os.getenv("RERANKER_BASE_URL", ""),
    api_key=os.getenv("RERANKER_API_KEY", ""),
    model_name=os.getenv("RERANKER_MODEL_NAME", "bge-reranker-large")
)
