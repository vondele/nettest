import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import yaml

from nettest import generate_pipeline


class TestMatchedRecipe(unittest.TestCase):
    def test_control_and_candidate_are_trained_independently(self):
        recipe_dir = Path(__file__).resolve().parents[1]
        for name in ("testing.yaml", "threats.yaml"):
            with self.subTest(recipe=name):
                with (recipe_dir / name).open() as file:
                    recipe = yaml.safe_load(file)

                def mark_new(expanded_recipe):
                    for step in expanded_recipe["training"]["steps"]:
                        step["status"] = "New"

                with patch.object(generate_pipeline, "workspace_status", mark_new):
                    with redirect_stdout(io.StringIO()):
                        pipeline, schedule, parsed = generate_pipeline.parse_recipe(recipe, None)

                c1, c2, k1, k2 = parsed["training"]["steps"]
                self.assertEqual(parsed["testing"]["reference"]["step-sha"], c2["sha"])
                self.assertEqual(schedule["test"][0]["testing_sha"], k2["sha"])
                self.assertEqual(len(schedule["test"]), 1)
                previous_by_step = {
                    task["current_sha"]: task["previous_sha"] for task in schedule["train"]
                }
                self.assertEqual(
                    previous_by_step,
                    {
                        c1["sha"]: "None",
                        c2["sha"]: c1["sha"],
                        k1["sha"]: "None",
                        k2["sha"]: k1["sha"],
                    },
                )

                match = pipeline[f"step_4_{k2['sha']}_test"]
                self.assertIn(f"step_2_{c2['sha']}_rep_0_train", match["needs"])
                self.assertIn(f"step_4_{k2['sha']}_rep_0_train", match["needs"])

                for control, candidate in ((c1, k1), (c2, k2)):
                    self.assertEqual(control["trainer"], candidate["trainer"])
                    for field in ("run", "convert"):
                        control_config = yaml.safe_dump(control[field])
                        self.assertEqual(
                            control_config.replace("K32H3^", "K20H3^"),
                            yaml.safe_dump(candidate[field]),
                        )


if __name__ == "__main__":
    unittest.main()
