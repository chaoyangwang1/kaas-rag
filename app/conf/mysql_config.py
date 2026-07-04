# 导入核心依赖：数据类、环境变量读取
from dataclasses import dataclass
import os
from dotenv import load_dotenv

# 提前加载.env配置文件（确保os.getenv能获取到MySQL相关配置）
load_dotenv()

# 定义MySQL数据库配置类（与MinIOConfig/MilvusConfig风格一致，字段对应.env配置项）
@dataclass
class MySQLConfig:
    host: str         # MySQL服务地址
    port: int         # MySQL服务端口
    user: str         # MySQL用户名
    password: str     # MySQL密码
    database: str     # 数据库名
    charset: str = "utf8mb4"  # 字符集，默认utf8mb4

# 实例化MySQL配置对象，自动从.env读取配置并绑定（带默认值兜底）
mysql_config = MySQLConfig(
    host=os.getenv("MYSQL_HOST", "127.0.0.1"),
    port=int(os.getenv("MYSQL_PORT", "3306")),
    user=os.getenv("MYSQL_USER", "root"),
    password=os.getenv("MYSQL_PASSWORD", ""),
    database=os.getenv("MYSQL_DATABASE", "kaas_rag"),
    charset=os.getenv("MYSQL_CHARSET", "utf8mb4"),
)
