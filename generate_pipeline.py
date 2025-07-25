import re
import yaml
import hashlib
import json
from pathlib import Path
from utils import MyDumper


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


def workspace_status(recipe, workspace_dir):
    """
    Ensure the workspace structure

    directories:
       workspace_dir / "scratch" / step["sha"] : a directory for each step of this job, maybe already computed by other jobs
       workspace_dir / "scratch" / testing["sha"] : a directory for testing nnues that result from this job

    files:
       workspace_dir / "scratch" / step["sha"] / step.yaml : the yaml description of this step
       workspace_dir / "scratch" / step["sha"] / final.yaml : a yaml description generated when the step is complete
       workspace_dir / "scratch" / testing["sha"] / testing.yaml : a yaml description of the testing stage

    """

    base_dir = Path(workspace_dir) / "scratch"
    base_dir.mkdir(parents=True, exist_ok=True)

    if "testing" in recipe:
        testing_dir = base_dir / recipe["testing"]["sha"]
        testing_dir.mkdir(parents=True, exist_ok=True)
        testing_yaml = testing_dir / "testing.yaml"
        with testing_yaml.open(mode="w", encoding="utf-8") as f:
            yaml.dump(
                recipe["testing"], f, Dumper=MyDumper, default_flow_style=False
            )

    if "training" not in recipe:
        return

    for step in recipe["training"]["steps"]:
        step_dir = base_dir / step["sha"]
        step_dir.mkdir(parents=True, exist_ok=True)
        final_status_file = step_dir / "final.yaml"
        if final_status_file.is_file():
            step["status"] = "Final"
        else:
            step["status"] = "New"  # TODO might need to deal with running steps etc.
            step_yaml = step_dir / "step.yaml"
            with step_yaml.open(mode="w", encoding="utf-8") as f:
                yaml.dump(step, f, Dumper=MyDumper, default_flow_style=False)

    return


def start_yaml():
    """
    Needed header for the CI pipeline
    """
    yaml_out = dict()
    yaml_out["include"] = [
        {
            "remote": "https://gitlab.com/cscs-ci/recipes/-/raw/master/templates/v2/.ci-ext.yml"
        }
    ]
    return yaml_out


def generate_stages(recipe, yaml_out):
    """
    List the needed stages in the recipe.
    Will skip training steps that are already in  Final stage.
    """
    stages = ["ensureData"]

    if "training" in recipe:
        for step in recipe["training"]["steps"]:
            if step["status"] == "Final":
                continue
            repetitions = step["run"].get("repetitions", 1)
            for rep in range(0, repetitions):
                stage_base_name = f"step_{step['sha']}"
                step["stage"] = stage_base_name
                stage_name = f"{stage_base_name}_{rep}"
                stages.append(stage_name)

    if "testing" in recipe:
        stages.append("testing")

    yaml_out["stages"] = stages

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


def generate_ensure_data(recipe, workspace_dir, yaml_out, shell_out):
    """
    Extract all datasets from the training steps, as a set of all needed Huggingface owner/repo tuples.
    """

    hfs = set()
    # Guarantee the presence of this one in all cases
    hfs.add(("official-stockfish", "master-binpacks"))

    # see what is needed, do not skip what is needed in finalized steps, might be useful to keep these datasets warm.
    if "training" in recipe:
        for step in recipe["training"]["steps"]:
            for dataset in step["datasets"]:
                hfs.add((dataset["hf"]["owner"], dataset["hf"]["repo"]))

    # actual job script steps..
    job = generate_job_base()
    job["variables"]["SLURM_TIMELIMIT"] = "04:00:00"
    job["stage"] = "ensureData"

    job["script"] = []
    for hf in hfs:
        job["script"].append(
            f"python -u {workspace_dir}/nettest/ensure_data.py {workspace_dir} {hf[0]} {hf[1]}"
        )

    shell_out += job["script"]

    yaml_out["ensureDataJob"] = job
    return


def generate_training_stages(
    recipe, workspace_dir, ci_project_dir, yaml_out, shell_out
):
    """
    Generate training stages, essentially just pointing out the current step sha and the one of the previous run.
    With this info (and the information saved on disk), the job should be able to execute.
    """
    if "training" not in recipe:
        return

    previous_sha = "None"

    for step in recipe["training"]["steps"]:
        if step["status"] == "Final":
            previous_sha = step["sha"]
            continue

        this_sha = step["sha"]

        repetitions = step["run"].get("repetitions", 1)
        for rep in range(0, repetitions):
            job = generate_job_base()

            stage_base_name = step["stage"]
            stage_name = f"{stage_base_name}_{rep}"
            job["stage"] = stage_name

            job["script"] = [
                f"python -u {workspace_dir}/nettest/train.py {workspace_dir} {this_sha} {previous_sha} {ci_project_dir}"
            ]

            shell_out += job["script"]

            job["artifacts"] = {"expire_in": "1 month", "paths": [f"step_{this_sha}"]}

            yaml_out[stage_name + "Job"] = job

        previous_sha = this_sha

    return


def generate_testing_stage(recipe, workspace_dir, yaml_out, shell_out):
    """
    Generate the testing stage
    """

    if "testing" not in recipe:
        return

    test_config_sha = recipe["testing"]["sha"]

    job = generate_job_base()
    job["stage"] = "testing"

    # pass the last training step sha as input, and all other steps that were computed in this run
    job["script"] = []
    steps = 0
    for step in reversed(recipe["training"]["steps"]):
        if step["status"] != "Final" or steps == 0:
            steps += 1
            step_sha = step["sha"]
            task = f"python -u {workspace_dir}/nettest/test.py {workspace_dir} {test_config_sha} {step_sha}"
            job["script"].append(task)

    shell_out += job["script"]
    yaml_out["testingJob"] = job

    return


def parse_recipe(recipe, workspace_dir, ci_project_dir):
    """
    Given recipe turn that recipe into a CI pipeline..
    """

    # ci yaml header
    yaml_out = start_yaml()
    shell_out = ["#!/bin/bash", "", "set -ex", ""]

    # insert shas that uniquely identify each step based on the full history of the training recipe
    insert_shas(recipe)

    # setup workspace, and figure out status
    workspace_status(recipe, workspace_dir)

    # generate the stage names / stages
    generate_stages(recipe, yaml_out)

    # generate the ensureData stages
    generate_ensure_data(recipe, workspace_dir, yaml_out, shell_out)

    # tricky bit ... generate the training stages
    generate_training_stages(
        recipe, workspace_dir, ci_project_dir, yaml_out, shell_out
    )

    # generate the match stage
    generate_testing_stage(recipe, workspace_dir, yaml_out, shell_out)

    return yaml_out, shell_out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Translate recipe to pipeline and shell script."
    )
    parser.add_argument("workspace_dir", help="Workspace directory")
    parser.add_argument("input_file", help="Input recipe file")
    parser.add_argument("output_file", help="Output pipeline YAML file")
    parser.add_argument("ci_project_dir", help="CI project directory")
    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file
    workspace_dir = args.workspace_dir
    ci_project_dir = args.ci_project_dir

    print("Translating recipe: ", input_file)

    with open(input_file) as f:
        recipe = yaml.safe_load(f)

    yaml_out, shell_out = parse_recipe(recipe, workspace_dir, ci_project_dir)

    print("Resulting pipeline: ", Path(output_file))
    with Path(output_file).open(mode="w", encoding="utf-8") as f:
        yaml.dump(yaml_out, f, Dumper=MyDumper, default_flow_style=False, width=300)

    print("Resulting shell script: ", Path(output_file).with_suffix(".sh"))
    with Path(output_file).with_suffix(".sh").open(mode="w", encoding="utf-8") as f:
        for line in shell_out:
            print(line, file=f)
