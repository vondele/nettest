import yaml

from .generate_pipeline import parse_recipe
from .ensure_data import run_data_update
from .train import run_step
from .test import run_test


def execute_local(recipe, workspace_dir, ci_project_dir):
    _, schedule = parse_recipe(recipe, workspace_dir, ci_project_dir)

    for args in schedule["data"]:
        run_data_update(**args)

    for args in schedule["train"]:
        run_step(**args)

    for args in schedule["test"]:
        run_test(**args)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Execute a recipe")
    parser.add_argument("workspace_dir", help="Workspace directory")
    parser.add_argument("input_file", help="Input recipe file")
    parser.add_argument("ci_project_dir", help="CI project directory")
    args = parser.parse_args()

    input_file = args.input_file
    workspace_dir = args.workspace_dir
    ci_project_dir = args.ci_project_dir

    print("Executing recipe: ", input_file)

    with open(input_file) as f:
        recipe = yaml.safe_load(f)

    execute_local(recipe, workspace_dir, ci_project_dir)
