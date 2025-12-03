import os
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.config import ClusterConfig, ClusterMode


class ClusterConfigTests(unittest.TestCase):
    def test_defaults_to_single_mode(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            cfg = ClusterConfig.from_env()
            self.assertEqual(cfg.mode, ClusterMode.SINGLE)

    def test_dual_mode_requires_ip(self) -> None:
        with patch.dict(os.environ, {"CLUSTER_MODE": "dual"}, clear=True):
            cfg = ClusterConfig.from_env()
            with self.assertRaises(ValueError):
                cfg.validate()

    def test_dataclass_defaults_load_from_env(self) -> None:
        with patch.dict(
            os.environ,
            {"CLUSTER_MODE": "dual", "SECOND_SPARK_IP": "10.10.0.42", "SECOND_SPARK_SSH_USER": "spark"},
            clear=True,
        ):
            cfg = ClusterConfig()
            self.assertEqual(cfg.mode, ClusterMode.DUAL)
            self.assertEqual(cfg.second_spark_ip, "10.10.0.42")
            self.assertEqual(cfg.second_spark_ssh_user, "spark")


if __name__ == "__main__":
    unittest.main()
