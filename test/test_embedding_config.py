"""
测试 Embedding 配置的可配置化：use_fp16 和 device
"""
import os
import sys
import importlib
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestEmbeddingConfigFields:
    """测试 EmbeddingConfig 数据类新增 use_fp16 和 device 字段"""

    def test_config_has_use_fp16_field(self):
        """use_fp16 字段存在，默认值为 True"""
        from app.conf.embedding_config import EmbeddingConfig

        config = EmbeddingConfig(
            base_url="http://test:8000/v1",
            api_key="test_key",
            model_name="bge-m3",
        )
        assert config.use_fp16 is True

    def test_config_has_device_field(self):
        """device 字段存在，默认值为 'cpu'"""
        from app.conf.embedding_config import EmbeddingConfig

        config = EmbeddingConfig(
            base_url="http://test:8000/v1",
            api_key="test_key",
            model_name="bge-m3",
        )
        assert config.device == "cpu"


class TestEmbeddingConfigFromEnv:
    """测试从环境变量读取 use_fp16 和 device 配置"""

    def test_use_fp16_from_env_true(self, monkeypatch):
        """EMBEDDING_USE_FP16=true → use_fp16=True"""
        monkeypatch.setenv("EMBEDDING_USE_FP16", "true")
        import app.conf.embedding_config as ec
        importlib.reload(ec)
        assert ec.embedding_config.use_fp16 is True

    def test_use_fp16_from_env_false(self, monkeypatch):
        """EMBEDDING_USE_FP16=false → use_fp16=False"""
        monkeypatch.setenv("EMBEDDING_USE_FP16", "false")
        import app.conf.embedding_config as ec
        importlib.reload(ec)
        assert ec.embedding_config.use_fp16 is False

    def test_device_default_cpu(self, monkeypatch):
        """未设置 EMBEDDING_DEVICE 时默认为 'cpu'"""
        import app.conf.embedding_config as ec
        importlib.reload(ec)
        assert ec.embedding_config.device == "cpu"

    def test_device_from_env_cuda(self, monkeypatch):
        """EMBEDDING_DEVICE=cuda → device='cuda'"""
        monkeypatch.setenv("EMBEDDING_DEVICE", "cuda")
        import app.conf.embedding_config as ec
        importlib.reload(ec)
        assert ec.embedding_config.device == "cuda"

    def test_device_from_env_cuda0(self, monkeypatch):
        """EMBEDDING_DEVICE=cuda:0 → device='cuda:0'"""
        monkeypatch.setenv("EMBEDDING_DEVICE", "cuda:0")
        import app.conf.embedding_config as ec
        importlib.reload(ec)
        assert ec.embedding_config.device == "cuda:0"


class TestLocalEmbeddingClientInit:
    """测试 LocalEmbeddingClient 将配置传递给 BGEM3FlagModel"""

    @patch("FlagEmbedding.BGEM3FlagModel")
    def test_passes_use_fp16_true(self, mock_bge):
        from app.lm.embedding_utils import LocalEmbeddingClient
        LocalEmbeddingClient(model_path="/fake/path", use_fp16=True, device="cpu")
        mock_bge.assert_called_once_with("/fake/path", use_fp16=True, devices="cpu")

    @patch("FlagEmbedding.BGEM3FlagModel")
    def test_passes_device_cuda0(self, mock_bge):
        from app.lm.embedding_utils import LocalEmbeddingClient
        LocalEmbeddingClient(model_path="/fake/path", use_fp16=True, device="cuda:0")
        mock_bge.assert_called_once_with("/fake/path", use_fp16=True, devices="cuda:0")

    @patch("FlagEmbedding.BGEM3FlagModel")
    def test_defaults(self, mock_bge):
        from app.lm.embedding_utils import LocalEmbeddingClient
        LocalEmbeddingClient(model_path="/fake/path")
        mock_bge.assert_called_once_with("/fake/path", use_fp16=True, devices="cpu")


class TestGetBgeM3WithConfig:
    """测试 get_bge_m3_local 和 get_bge_m3_ef 使用 device 配置"""

    @patch("FlagEmbedding.BGEM3FlagModel")
    def test_get_bge_m3_local_uses_device_config(self, mock_bge):
        """get_bge_m3_local() 将 embedding_config.device 传给 BGEM3FlagModel"""
        import app.conf.embedding_config as ec
        import app.lm.embedding_utils as eu
        importlib.reload(ec)
        importlib.reload(eu)

        eu._bge_m3_local = None
        ec.embedding_config.device = "cuda:1"
        ec.embedding_config.use_fp16 = False
        ec.embedding_config.local_model_path = "/test/model"

        eu.get_bge_m3_local()
        mock_bge.assert_called_with("/test/model", use_fp16=False, devices="cuda:1")

    @patch("FlagEmbedding.BGEM3FlagModel")
    def test_get_bge_m3_local_device_cpu_default(self, mock_bge):
        """device 默认值 'cpu' 正确传递"""
        import app.conf.embedding_config as ec
        import app.lm.embedding_utils as eu
        importlib.reload(ec)
        importlib.reload(eu)

        eu._bge_m3_local = None
        ec.embedding_config.local_model_path = "/test/model"

        eu.get_bge_m3_local()
        _, kwargs = mock_bge.call_args
        assert kwargs.get("devices") == "cpu"
