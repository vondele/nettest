import unittest
import copy
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from nettest.meta_recipe import intelligent_merge

class TestIntelligentMerge(unittest.TestCase):
    def setUp(self):
        self.base_dict = {
            "run": {
                "binpacks": ["a.bin", "b.bin", "c.bin", "d.bin"],
                "other_options": ["--lr=0.1", "--gamma=0.9", "--ft_optimize"]
            }
        }

    def test_extend_no_slice(self):
        override = {"run": {"<extend_bp>": {"binpacks": ["e.bin"]}}}
        result = intelligent_merge(self.base_dict, override)
        self.assertEqual(result["run"]["binpacks"], ["a.bin", "b.bin", "c.bin", "d.bin", "e.bin"])

    def test_extend_start_slice(self):
        # Drop the first 2 items, then append
        override = {"run": {"<extend(2,)>": {"binpacks": ["e.bin"]}}}
        result = intelligent_merge(self.base_dict, override)
        self.assertEqual(result["run"]["binpacks"], ["c.bin", "d.bin", "e.bin"])

    def test_extend_end_slice(self):
        # Drop the last item, then append
        override = {"run": {"<extend(,-1)>": {"binpacks": ["e.bin"]}}}
        result = intelligent_merge(self.base_dict, override)
        self.assertEqual(result["run"]["binpacks"], ["a.bin", "b.bin", "c.bin", "e.bin"])

    def test_extend_both_slices(self):
        # Keep only the middle elements (index 1 to 3 exclusive)
        override = {"run": {"<extend(1,3)>": {"binpacks": ["e.bin"]}}}
        result = intelligent_merge(self.base_dict, override)
        self.assertEqual(result["run"]["binpacks"], ["b.bin", "c.bin", "e.bin"])

    def test_extend_missing_base_key(self):
        # Should cleanly create the list if it doesn't exist
        override = {"run": {"<extend>": {"new_list": ["a", "b"]}}}
        result = intelligent_merge(self.base_dict, override)
        self.assertEqual(result["run"]["new_list"], ["a", "b"])

    def test_intelligent_cli_args_merge(self):
        # Should update --lr, leave --gamma alone, add --new-flag, keep --ft_optimize
        override = {"run": {"other_options": ["--lr=0.01", "--new-flag"]}}
        result = intelligent_merge(self.base_dict, override)
        self.assertIn("--lr=0.01", result["run"]["other_options"])
        self.assertNotIn("--lr=0.1", result["run"]["other_options"])
        self.assertIn("--gamma=0.9", result["run"]["other_options"])
        self.assertIn("--ft_optimize", result["run"]["other_options"])
        self.assertIn("--new-flag", result["run"]["other_options"])

    def test_explicit_replace(self):
        # Should wipe the list completely and insert only the new items
        override = {"run": {"<replace_opts>": {"other_options": ["--lr=0.5"]}}}
        result = intelligent_merge(self.base_dict, override)
        self.assertEqual(result["run"]["other_options"], ["--lr=0.5"])

if __name__ == '__main__':
    unittest.main()