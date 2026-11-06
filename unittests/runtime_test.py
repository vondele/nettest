import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nettest.runtime import runtime_command


class TestMatchedRuntime(unittest.TestCase):
    def test_visible_gpu_mapping_is_shared_by_export_and_crosscheck(self):
        with patch.dict("os.environ", {"TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "1"}):
            args, env = runtime_command(
                {"train": {"devices": "2,5,"}},
                {"device": "auto", "matmul-precision": "highest"},
            )
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "2,5")
        self.assertNotIn("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", env)
        self.assertEqual(args, ["--device=auto", "--matmul-precision=highest"])

    def test_explicit_cpu_and_numeric_device_alias(self):
        self.assertIn("--device=cpu", runtime_command({}, {"device": "cpu"})[0])
        self.assertIn("--device=cuda:1", runtime_command({}, {"device": 1})[0])
