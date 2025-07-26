import yaml
from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED
from firecrest_executor import FirecrestExecutor

from .generate_pipeline import parse_recipe
from .ensure_data import run_data_update
from .train import run_step
from .test import run_test

# useful to batch a number of calls,
# needs to pass the function so it can be found remotely
# to avoid (attempted relative import with no known parent package)
def batch_function(f, items):
    for kwargs in items:
        f(**kwargs)

def execute(executor, recipe):
    _, schedule = executor.submit(parse_recipe, recipe).result()

    executor.submit(batch_function, run_data_update, schedule["data"]).result()

    for kwargs in schedule["train"]:
        executor.submit(run_step, **kwargs).result()

    # do parallel tests, if the executor supports it
    futures = []
    for kwargs in schedule["test"]:
        futures.append(executor.submit(run_test, **kwargs))

    done, _ = wait(futures, return_when=ALL_COMPLETED)

    # if there are multiple tests, we only return the highest Elo
    Elo = None
    for future in done:
        _, result = future.result()
        if Elo is None or result > Elo:
            Elo = result

    return Elo


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Execute a recipe")
    parser.add_argument(
        "--executor",
        choices=["local", "remote"],
        default="local",
        help="Executor type: local or remote",
    )
    parser.add_argument("--recipe", required=True, help="Input recipe file")
    args = parser.parse_args()

    print("Executing recipe: ", args.recipe)
    with open(args.recipe) as f:
        recipe = yaml.safe_load(f)

    if args.executor == "local":
        executor = ProcessPoolExecutor(max_workers=1)
    else:
        executor = FirecrestExecutor(
            working_dir="/users/vjoost/fish/workspace/",
            sbatch_options=[
                "--job-name=FirecrestExecutor",
                "--time=12:00:00",
                "--nodes=1",
                "--partition=normal",
            ],
            srun_options=["--environment=/users/vjoost/fish/workspace/nettest.toml"],
            sleep_interval=5,
            max_workers=64,
        )

    Elo = execute(executor, recipe)
    print(f"Execution of the recipe led to a net of {Elo} Elo.")

    executor.shutdown(wait=True, cancel_futures=False)
