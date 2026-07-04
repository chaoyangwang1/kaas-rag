# 导入核心依赖：数据类、环境变量读取、路径处理
from dataclasses import dataclass
import os
from dotenv import load_dotenv

# 提前加载.env配置文件
load_dotenv()

# 定义Embedding远程服务配置
@dataclass
class EmbeddingConfig:
    base_url: str      # 远程服务地址，如 http://10.0.0.5:8000/v1
    api_key: str       # API密钥，不需要认证则留空
    model_name: str    # 模型名称，如 bge-m3
    use_local_model: bool = False     # 是否使用本地模型
    local_model_path: str = ""        # 本地模型路径，如 F:/docker/models/bge-m3
    use_fp16: bool = True             # 本地模型是否使用 FP16 半精度（节省内存）
    device: str = "cpu"              # 本地模型运行设备，如 cpu / cuda / cuda:0 / cuda:1

# 实例化配置对象
embedding_config = EmbeddingConfig(
    base_url=os.getenv("EMBEDDING_BASE_URL", ""),
    api_key=os.getenv("EMBEDDING_API_KEY", ""),
    model_name=os.getenv("EMBEDDING_MODEL_NAME", "bge-m3"),
    use_local_model=os.getenv("EMBEDDING_USE_LOCAL", "false").lower() in ("true", "1", "yes"),
    local_model_path=os.getenv("EMBEDDING_LOCAL_MODEL_PATH", ""),
    use_fp16=os.getenv("EMBEDDING_USE_FP16", "true").lower() in ("true", "1", "yes"),
    device=os.getenv("EMBEDDING_DEVICE", "cpu"),
)
