import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nettest.runtime import runtime_command
from nettest.test import run_cross_check_eval
from nettest.train import run_conversion, run_trainer


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

    def test_runtime_propagates_through_training_conversion_and_crosscheck(self):
        environment = {"train": {"devices": "2,5,"}}
        settings = {"device": "cuda:0", "matmul-precision": "high"}
        with (
            TemporaryDirectory() as tmp,
            patch.dict("os.environ", {"TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "1"}),
        ):
            root = Path(tmp)
            run_dir = root / "scratch" / "step-id" / "run"
            run_dir.mkdir(parents=True)
            checkpoint = run_dir / "last.ckpt"
            checkpoint.touch()
            data = root / "data"
            data.mkdir()
            (data / "positions.binpack").touch()
            (root / "cidir").mkdir()
            engine = root / "stockfish"
            engine.touch()
            step_dir = run_dir.parent
            (step_dir / "step.yaml").write_text("trainer: {}\n")

            def serialize(_name, command, *_args, **_kwargs):
                Path(command[4]).write_bytes(b"test artifact")

            with (
                patch.object(Path, "cwd", return_value=root),
                patch("nettest.train.find_most_recent", return_value=checkpoint),
                patch("nettest.train.ckpt_reached_end", side_effect=[False, True]),
                patch("nettest.train.execute", side_effect=serialize) as conversion,
                patch("nettest.test.ensure_trainer", return_value=root),
                patch("nettest.test.execute") as crosscheck,
            ):
                # Force a new step while reusing the fixture checkpoint when the
                # launcher checks for output after the stubbed training command.
                with patch("nettest.train.execute") as training:
                    self.assertTrue(
                        run_trainer(
                            environment,
                            "step-id",
                            None,
                            {
                                "binpacks": ["positions.binpack"],
                                "max_epochs": 1,
                                "other_options": {"matmul-precision": "high"},
                            },
                            root,
                        )
                    )
                run_conversion(
                    environment,
                    "step-id",
                    {
                        "binpack": "positions.binpack",
                        "checkpoint2nnue": {},
                        "optimize": {"ft-optimize": True},
                        "runtime": settings,
                    },
                    root,
                )
                final = yaml.safe_load((step_dir / "final.yaml").read_text())
                self.assertEqual(final["runtime"], settings)
                run_cross_check_eval(
                    environment,
                    {"crosscheck": {"binpack": "positions.binpack"}},
                    "step-id",
                    engine,
                )

            self.assertEqual(conversion.call_count, 3)
            crosscheck.assert_called_once()
            for call in [
                training.call_args,
                *conversion.call_args_list,
                crosscheck.call_args,
            ]:
                self.assertIn("--matmul-precision=high", call.args[1])
                self.assertEqual(call.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "2,5")
                self.assertNotIn("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", call.kwargs["env"])
            for call in [*conversion.call_args_list, crosscheck.call_args]:
                self.assertIn("--device=cuda:0", call.args[1])
