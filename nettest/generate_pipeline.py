import re
import yaml
import hashlib
import json
import argparse
from pathlib import Path
from collections import defaultdict
from .utils import MyDumper
from .meta_recipe import expand_meta_recipe


def needs_quotes(value):
    # You can tweak this to match other patterns too
    return isinstance(value, str) and re.match(r"^\d{1,2}:\d{2}:\d{2}$", value)


def quoted_scalar_representer(dumper, data):
    style = '"' if needs_quotes(data) else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


yaml.add_representer(str, quoted_scalar_representer, Dumper=MyDumper)


def insert_shas(recipe):
    """
    1. Insert in each training step a sha that unique identifies it for later reuse.
       It is based on the shas of preceding steps (assume some restart), and the content of the step.
    2. Insert a sha in the testing phase
    """
    if "training" in recipe:
        stepHash = ""
        for step in recipe["training"]["steps"]:
            step_content = json.dumps(step, sort_keys=True)
            combined = stepHash + step_content
            stepHash = hashlib.sha256(combined.encode("utf-8")).hexdigest()[:12]
            step["sha"] = stepHash

    if "testing" in recipe:
        stepHash = ""
        step = recipe["testing"]
        step_content = json.dumps(step, sort_keys=True)
        combined = stepHash + step_content
        stepHash = hashlib.sha256(combined.encode("utf-8")).hexdigest()[:12]
        step["sha"] = stepHash

    return


def workspace_status(recipe):
    """
    Ensure the workspace structure

    directories:
       cwd / "scratch" / step["sha"] : a directory for each step of this job, maybe already computed by other jobs
       cwd / "scratch" / testing["sha"] : a directory for testing nnues that result from this job

    files:
       cwd / "scratch" / step["sha"] / step.yaml : the yaml description of this step
       cwd / "scratch" / step["sha"] / final.yaml : a yaml description generated when the step is complete
       cwd / "scratch" / testing["sha"] / testing.yaml : a yaml description of the testing stage

    """

    workspace_dir = Path.cwd()

    base_dir = Path(workspace_dir) / "scratch"
    base_dir.mkdir(parents=True, exist_ok=True)

    if "testing" in recipe:
        testing_dir = base_dir / recipe["testing"]["sha"]
        testing_dir.mkdir(parents=True, exist_ok=True)
        testing_yaml = testing_dir / "testing.yaml"
        with testing_yaml.open(mode="w", encoding="utf-8") as f:
            yaml.dump(recipe["testing"], f, Dumper=MyDumper, default_flow_style=False)

    if "training" not in recipe:
        return

    for step in recipe["training"]["steps"]:
        step_dir = base_dir / step["sha"]
        step_dir.mkdir(parents=True, exist_ok=True)
        final_status_file = step_dir / "final.yaml"
        if final_status_file.is_file():
            step["status"] = "Final"
            with open(final_status_file) as f:
                final_info = yaml.safe_load(f)
                step["std_nnue"] = final_info["std_nnue"]
        else:
            step["status"] = "New"  # TODO might need to deal with running steps etc.
            step_yaml = step_dir / "step.yaml"
            with step_yaml.open(mode="w", encoding="utf-8") as f:
                yaml.dump(step, f, Dumper=MyDumper, default_flow_style=False)

    return


def start_ci_yaml():
    """
    Needed header for the CI pipeline
    """
    ci_yaml_out = dict()
    ci_yaml_out["include"] = [
        {
            "remote": "https://gitlab.com/cscs-ci/recipes/-/raw/master/templates/v2/.ci-ext.yml"
        }
    ]
    return ci_yaml_out


def generate_stages(recipe, ci_yaml_out):
    """
    List the needed stages in the recipe.
    Will skip training steps that are already in  Final stage.
    """
    stages = ["ensureData"]

    if "training" in recipe:
        step_number = 0
        for step in recipe["training"]["steps"]:
            step_number += 1
            if step["status"] == "Final":
                continue
            repetitions = step["run"].get("repetitions", 1)
            for rep in range(0, repetitions):
                stage_base_name = f"step_{step_number}_{step['sha']}"
                step["stage"] = stage_base_name
                stage_name = f"{stage_base_name}_rep_{rep}"
                stages.append(stage_name)

    if "testing" in recipe:
        stages.append("testing")

    ci_yaml_out["stages"] = stages

    return


def generate_job_base():
    """
    Generate a base ci yaml setup for a computational job
    """
    variables = {
        "SLURM_JOB_NUM_NODES": 1,
        "SLURM_NTASKS": 1,
        "SLURM_TIMELIMIT": "12:00:00",
        "SLURM_CPU_BIND": "none",
    }
    job = {
        "timeout": "48h",
        "extends": ".container-runner-clariden-gh200",
        "image": "$PERSIST_IMAGE_NAME",
        "variables": variables,
    }

    return job


def generate_ensure_data(recipe, ci_yaml_out, schedule):
    """
    Extract all datasets from the training steps
    """

    # collect all binpacks that are needed
    binpacks = set()
    if "training" in recipe:
        for step in recipe["training"]["steps"]:
            if "convert" in step and "binpack" in step["convert"]:
                binpacks.add(step["convert"]["binpack"])
            if "run" in step and "binpacks" in step["run"]:
                for binpack in step["run"]["binpacks"]:
                    binpacks.add(binpack)

    repos = defaultdict(list)
    for binpack in binpacks:
        owner, repo, filename = binpack.split("/", 2)
        repos[(owner, repo)].append(filename)

    # actual job script steps..
    job = generate_job_base()
    job["variables"]["SLURM_TIMELIMIT"] = "04:00:00"
    job["stage"] = "ensureData"

    job["script"] = ["cd /workspace", "ln -s $CI_PROJECT_DIR ./cidir"]
    for (owner, repo), filenames in repos.items():
        job["script"].append(
            f"python -u -m nettest.ensure_data {owner} {repo} " + " ".join(filenames)
        )
        schedule["data"].append({"owner": owner, "repo": repo, "filenames": filenames})

    ci_yaml_out["ensureDataJob"] = job
    return


def generate_training_stages(recipe, environment, ci_yaml_out, schedule):
    """
    Generate training stages, essentially just pointing out the current step sha and the one of the previous run.
    With this info (and the information saved on disk), the job should be able to execute.
    """
    training_jobs_by_sha = defaultdict(list)

    if "training" not in recipe:
        return training_jobs_by_sha

    previous_sha = "None"

    envarg = f"--environment {environment}" if environment else ""
    step_number = 0

    for step in recipe["training"]["steps"]:
        step_number += 1
        current_sha = step["sha"]
        print(
            f"Step {step_number} : starting from {previous_sha} leading to {current_sha}",
            flush=True,
        )

        if step["status"] == "Final":
            std_nnue = step.get("std_nnue", "unknown")
            print(
                f"--> step {step_number} is final already. Result: {std_nnue}",
                flush=True,
            )
            previous_sha = current_sha
            continue

        max_epochs = step["run"].get("max_epochs", 0)
        print(f"--> step {step_number} needs {max_epochs} epochs", flush=True)

        repetitions = step["run"].get("repetitions", 1)
        for rep in range(0, repetitions):
            job = generate_job_base()

            stage_base_name = step["stage"]
            stage_name = f"{stage_base_name}_rep_{rep}"
            job["stage"] = stage_name

            job["script"] = [
                "cd /workspace/",
                "ln -s $CI_PROJECT_DIR ./cidir",
                f"python -u -m nettest.train {envarg} {current_sha} {previous_sha}",
            ]

            schedule["train"].append(
                {
                    "current_sha": current_sha,
                    "previous_sha": previous_sha,
                }
            )

            job["artifacts"] = {
                "expire_in": "1 month",
                "paths": [f"step_{current_sha}"],
            }

            job_name = stage_name + "_train"
            ci_yaml_out[job_name] = job
            training_jobs_by_sha[current_sha].append(job_name)

        previous_sha = current_sha

    return training_jobs_by_sha


def generate_testing_stage(
    recipe, environment, ci_yaml_out, schedule, training_jobs_by_sha
):
    """
    Generate the testing stage
    """

    if "testing" not in recipe:
        return

    test_config_sha = recipe["testing"]["sha"]
    test_steps = recipe["testing"].get("steps", "new")
    assert test_steps in ["new", "all", "last"], (
        "testing steps must be 'new', 'all' or 'last'"
    )

    envarg = f"--environment {environment}" if environment else ""

    # pass the last training step sha as input, and all other steps that were computed in this run
    steps = 0
    step_number = len(recipe["training"]["steps"])
    for step in reversed(recipe["training"]["steps"]):
        use_step = False
        if test_steps == "all":
            use_step = True
        elif test_steps == "new" and (step["status"] != "Final" or steps == 0):
            use_step = True
        elif test_steps == "last" and steps == 0:
            use_step = True

        if use_step:
            steps += 1
            testing_sha = step["sha"]
            task = f"python -u -m nettest.test {envarg} {test_config_sha} {testing_sha}"

            job = generate_job_base()
            job["stage"] = "testing"
            job["script"] = ["cd /workspace/", "ln -s $CI_PROJECT_DIR ./cidir"]
            job["script"].append(task)

            # Establish explicit DAG dependencies to bypass stage wait times
            needs = []
            if "ensureDataJob" in ci_yaml_out:
                needs.append("ensureDataJob")

            if testing_sha in training_jobs_by_sha:
                needs.extend(training_jobs_by_sha[testing_sha])

            if needs:
                job["needs"] = needs

            ci_yaml_out[f"step_{step_number}_{testing_sha}_test"] = job

            schedule["test"].append(
                {
                    "test_config_sha": test_config_sha,
                    "testing_sha": testing_sha,
                }
            )

        step_number -= 1

    return


def parse_recipe(recipe, environment):
    """
    Given recipe turn that recipe into a CI pipeline..
    """

    # ci yaml header
    ci_yaml_out = start_ci_yaml()
    schedule = {"data": [], "train": [], "test": []}

    # expand the meta recipe, i.e. handle the <repeat_last> directives and apply the overrides in the training steps sequentially.
    recipe = expand_meta_recipe(recipe)

    # insert shas that uniquely identify each step based on the full history of the training recipe
    insert_shas(recipe)

    print("Recipe, augmented with shas:")
    print(yaml.dump(recipe, Dumper=MyDumper, default_flow_style=False, width=300))

    # setup workspace, and figure out status
    workspace_status(recipe)

    # generate the stage names / stages
    generate_stages(recipe, ci_yaml_out)

    # generate the ensureData stages
    generate_ensure_data(recipe, ci_yaml_out, schedule)

    # generate the training stages and capture job dependencies
    training_jobs_by_sha = generate_training_stages(
        recipe, environment, ci_yaml_out, schedule
    )

    # generate the match stage utilizing the extracted job dependencies
    generate_testing_stage(
        recipe, environment, ci_yaml_out, schedule, training_jobs_by_sha
    )

    print("schedule information:")
    print(yaml.dump(schedule, Dumper=MyDumper, default_flow_style=False, width=300))

    return ci_yaml_out, schedule


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Translate recipe to pipeline and schedule"
    )
    parser.add_argument(
        "--environment", required=False, help="Definition of the environment file"
    )
    # Now accepts a colon-separated list of names, e.g., /path/to/test1:test2
    parser.add_argument("input_files", help="Input recipe names (colon-separated)")
    parser.add_argument("output_file", help="Output pipeline YAML file")
    args = parser.parse_args()

    # 1. Resolve input paths and shared directory
    input_parts = [p.strip() for p in args.input_files.split(":")]
    first_input = Path(input_parts[0])
    base_dir = first_input.parent

    recipe_paths = []
    for i, name in enumerate(input_parts):
        # Apply the directory from the first entry to all subsequent names
        p = first_input if i == 0 else base_dir / name
        recipe_paths.append(p.with_suffix(".yaml"))

    # 2. Global structures to hold the merged pipeline
    final_ci_out = {"include": [], "stages": []}
    final_schedule = {"data": [], "train": [], "test": []}

    merged_ensure_job = None
    merged_python_lines = set()
    global_stages = []

    for recipe_path in recipe_paths:
        recipe_stem = recipe_path.stem
        if not recipe_path.exists():
            print(f"Warning: {recipe_path} not found. Skipping.")
            continue

        print(f"Translating recipe: {recipe_path}")
        with open(recipe_path) as f:
            recipe = yaml.safe_load(f)

        # Generate the specific CI dict for this recipe
        ci_out, schedule = parse_recipe(recipe, args.environment)

        # Merge Includes (unique)
        for inc in ci_out.get("include", []):
            if inc not in final_ci_out["include"]:
                final_ci_out["include"].append(inc)

        # Merge Stages (tracking unique names)
        for stage in ci_out.get("stages", []):
            if stage not in global_stages:
                global_stages.append(stage)

        # Merge Schedule (sequential for shell case)
        for key in ["data", "train", "test"]:
            final_schedule[key].extend(schedule.get(key, []))

        # Handle the ensureDataJob
        if "ensureDataJob" in ci_out:
            if merged_ensure_job is None:
                # Extract template from the first recipe's data job
                merged_ensure_job = ci_out["ensureDataJob"].copy()
                # Keep only non-python setup lines (cd, ln, etc.)
                merged_ensure_job["script"] = [
                    line
                    for line in ci_out["ensureDataJob"]["script"]
                    if not line.strip().startswith("python")
                ]

            # Collect all unique python data commands
            for line in ci_out["ensureDataJob"]["script"]:
                if line.strip().startswith("python"):
                    merged_python_lines.add(line)

        # Merge and prefix all other jobs for concurrency
        for job_name, job_body in ci_out.items():
            if job_name in ["include", "stages", "ensureDataJob"]:
                continue

            prefixed_name = f"{recipe_stem}_{job_name}"

            # Update dependencies: ensureDataJob stays global, others get prefixed
            if "needs" in job_body:
                job_body["needs"] = [
                    need if need == "ensureDataJob" else f"{recipe_stem}_{need}"
                    for need in job_body["needs"]
                ]

            final_ci_out[prefixed_name] = job_body

    # 3. Fix Stage Ordering to avoid GitLab "need is not defined" errors
    # ensureData must be first, testing must be last.
    if "ensureData" in global_stages:
        global_stages.remove("ensureData")
        global_stages.insert(0, "ensureData")
    if "testing" in global_stages:
        global_stages.remove("testing")
        global_stages.append("testing")
    final_ci_out["stages"] = global_stages

    # 4. Finalize the merged data job
    if merged_ensure_job:
        merged_ensure_job["script"].extend(sorted(list(merged_python_lines)))
        final_ci_out["ensureDataJob"] = merged_ensure_job

    print(f"Resulting pipeline: {Path(args.output_file)}")
    with Path(args.output_file).open(mode="w", encoding="utf-8") as f:
        yaml.dump(final_ci_out, f, Dumper=MyDumper, default_flow_style=False, width=300)

    print("Final schedule information:")
    print(
        yaml.dump(final_schedule, Dumper=MyDumper, default_flow_style=False, width=300)
    )
