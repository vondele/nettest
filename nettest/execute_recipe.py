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


def execute(executor, recipe, environment):
    _, schedule = executor.submit(parse_recipe, recipe).result()

    print("submitting data update", flush=True)
    executor.submit(batch_function, run_data_update, schedule["data"]).result()

    itrain = 0
    ntrain = len(schedule["train"])
    for kwargs in schedule["train"]:
        itrain += 1
        print(f"submitting training step {itrain} / {ntrain}", flush=True)
        executor.submit(run_step, environment, **kwargs).result()

    # do parallel tests, if the executor supports it
    futures = []
    itest = 0
    ntest = len(schedule["test"])
    for kwargs in schedule["test"]:
        print(f"submitting testing step {itest} / {ntest}", flush=True)
        futures.append(executor.submit(run_test, environment, **kwargs))

    done, _ = wait(futures, return_when=ALL_COMPLETED)

    # if there are multiple tests, we only return the highest Elo
    nElo = None
    for future in done:
        _, result = future.result()
        if nElo is None or result > nElo:
            nElo = result

    print("all done", flush=True)

    return nElo


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
    parser.add_argument(
        "--environment", required=False, help="Definition of the environment file"
    )
    args = parser.parse_args()

    print("Executing recipe: ", args.recipe)
    with open(args.recipe) as f:
        recipe = yaml.safe_load(f)

    if args.environment:
        print("Using environment file: ", args.environment)
        with open(args.environment) as f:
            environment = yaml.safe_load(f)
    else:
        environment = dict()

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

    nElo = execute(executor, recipe, environment)
    print(f"Execution of the recipe led to a net of {nElo} nElo.")

    executor.shutdown(wait=True, cancel_futures=False)
