import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nettest.train import run_step


class TestRoutingDiagnostics(unittest.TestCase):
    def test_publish_for_finished_interrupted_and_failed_training(self):
        for outcome in (True, False, RuntimeError("test failure")):
            with self.subTest(outcome=outcome), TemporaryDirectory() as tmp:
                root = Path(tmp)
                stage = root / "scratch" / "job"
                run = stage / "run"
                run.mkdir(parents=True)
                (root / "cidir").mkdir()
                (stage / "step.yaml").write_text(
                    "sha: job\ntrainer: {}\nrun: {}\nconvert: {routing_diagnostics: true}\n"
                )
                paths = ["routing-settings.json", "routing-history.csv"]
                if outcome is True:
                    paths.append("final-routing.csv")
                for name in paths:
                    (run / name).write_text(f"{name} contents")
                (run / "last.ckpt").touch()
                (run / "last.pt").touch()

                with (
                    patch.object(Path, "cwd", return_value=root),
                    patch("nettest.train.ensure_trainer", return_value=root),
                    patch(
                        "nettest.train.run_trainer",
                        return_value=outcome,
                        side_effect=outcome if isinstance(outcome, Exception) else None,
                    ),
                    patch("nettest.train.run_conversion") as convert,
                ):
                    if isinstance(outcome, Exception):
                        with self.assertRaisesRegex(RuntimeError, "test failure"):
                            run_step({}, "job", "None")
                    else:
                        run_step({}, "job", "None")
                self.assertEqual(convert.call_count, int(outcome is True))
                artifacts = root / "cidir" / "step_job"
                self.assertEqual(
                    sorted(p.name for p in artifacts.iterdir()), sorted(paths)
                )
                for name in paths:
                    self.assertEqual((artifacts / name).read_text(), f"{name} contents")
