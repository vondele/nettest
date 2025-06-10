import sys
import yaml
import hashlib
import json
from pprint import pprint
from pathlib import Path

class MyDumper(yaml.Dumper):
    """
    Adjust yaml output to what is expected in gitlab CI...
    """

    def increase_indent(self, flow=False, indentless=False):
        return super(MyDumper, self).increase_indent(flow, False)


def insert_shas(procedure):
    """
    Insert in each training step a sha that unique identifies it for later reuse.
    It is based on the shas of preceding steps (assume some restart), and the content of the step.
    """
    if not "training" in procedure:
        return

    stepHash = ""
    for step in procedure["training"]["steps"]:
        step_content = json.dumps(step, sort_keys=True)
        combined = stepHash + step_content
        stepHash = hashlib.sha256(combined.encode("utf-8")).hexdigest()[:12]
        step["sha"] = stepHash

    return


def workspace_status(procedure, workspace_dir, ci_commit_sha):
    """
    Ensure the workspace structure

    directories:
       workspace_dir / ci_commit_sha : a directory for this particular CI job
       workspace_dir / step["sha"] : a directory for each step of this job, maybe already computed by other jobs

    files:
       workspace_dir / step["sha"] / step.yaml : the yaml description of this step
       workspace_dir / step["sha"] / final.yaml : a yaml description generated when the step is complete
       workspace_dir / ci_commit_sha / testing / testing.yaml : a yaml description of the testing stage

    """

    base_dir = Path(workspace_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    commit_dir = base_dir / ci_commit_sha
    commit_dir.mkdir(parents=True, exist_ok=True)

    if "testing" in procedure:
        testing_dir = commit_dir / "testing"
        testing_dir.mkdir(parents=True, exist_ok=True)
        testing_yaml = testing_dir / "testing.yaml"
        with testing_yaml.open(mode="w", encoding="utf-8") as f:
            yaml.dump(
                procedure["testing"], f, Dumper=MyDumper, default_flow_style=False
            )

    if not "training" in procedure:
        return

    for step in procedure["training"]["steps"]:
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


def generate_stages(procedure, yaml_out):
    """
    List the needed stages in the procedure.
    Will skip training steps that are already in  Final stage.
    """
    stages = ["ensureData"]

    if "training" in procedure:
        for step in procedure["training"]["steps"]:
            if step["status"] == "Final":
                continue
            stage_name = f"step_{step['sha']}"
            step["stage"] = stage_name
            stages.append(stage_name)

    if "testing" in procedure:
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
    }
    job = {
        "timeout": "48h",
        "extends": ".container-runner-clariden-gh200",
        "image": "$PERSIST_IMAGE_NAME",
        "variables": variables,
    }

    return job


def generate_ensure_data(procedure, yaml_out):
    """
    Extract all datasets from the training steps, as a set of all needed Huggingface owner/repo tuples.
    """

    hfs = set()
    # Guarantee the presence of this one in all cases
    hfs.add(("official-stockfish", "master-binpacks"))

    # see what is needed, do not skip what is needed in finalized steps, might be useful to keep these datasets warm.
    if "training" in procedure:
        for step in procedure["training"]["steps"]:
            for dataset in step["datasets"]:
                hfs.add((dataset["hf"]["owner"], dataset["hf"]["repo"]))

    # actual job script steps..
    job = generate_job_base()
    job["variables"]["SLURM_TIMELIMIT"] = "04:00:00"
    job["stage"] = "ensureData"

    job["script"] = []
    for hf in hfs:
        job["script"].append(f"/workspace/nettest/do_ensure_data.sh {hf[0]} {hf[1]}")

    yaml_out["ensureDataJob"] = job
    return


def generate_training_stages(procedure, workspace_dir, yaml_out):
    """
    Generate training stages, essentially just pointing out the current step sha and the one of the previous run.
    With this info (and the information saved on disk), the job should be able to execute.
    """
    if not "training" in procedure:
        return

    previous_sha = "None"

    for step in procedure["training"]["steps"]:
        if step["status"] == "Final":
            previous_sha = step["sha"]
            continue

        this_sha = step["sha"]

        job = generate_job_base()

        stage_name = step["stage"]
        job["stage"] = stage_name

        job["script"] = [
            f"python /workspace/nettest/do_step.py {this_sha} {previous_sha} {workspace_dir}"
        ]
        yaml_out[stage_name + "Job"] = job

        previous_sha = this_sha

    return


def generate_testing_stage(procedure, yaml_out):
    """
    Generate the testing stage
    """

    if not "testing" in procedure:
        return

    job = generate_job_base()
    job["stage"] = "testing"

    # pass the last training step sha as input
    if "training" in procedure:
        previous_sha = procedure["training"]["steps"][-1]["sha"]
    else:
        previous_sha = "None"

    # TODO
    job["script"] = [f"python /workspace/nettest/do_testing.py {previous_sha}"]

    yaml_out["testingJob"] = job

    return


def parse_procedure(input_path, workspace_dir, ci_commit_sha):
    """
    Given a file path, open that yaml, and turn that procedure into a CI pipeline..
    """

    with open(input_path) as f:
        procedure = yaml.safe_load(f)

    # ci yaml header
    yaml_out = start_yaml()

    # insert shas that uniquely identify each step based on the full history of the training procedure
    insert_shas(procedure)

    # setup workspace, and figure out status
    workspace_status(procedure, workspace_dir, ci_commit_sha)

    # generate the stage names / stages
    generate_stages(procedure, yaml_out)

    # generate the ensureData stages
    generate_ensure_data(procedure, yaml_out)

    # tricky bit ... generate the training stages
    generate_training_stages(procedure, workspace_dir, yaml_out)

    # generate the match stage
    generate_testing_stage(procedure, yaml_out)

    return yaml_out


if __name__ == "__main__":

    if len(sys.argv) != 5:
        print(
            "Usage: python do_generate_yaml_schedule.py input_file output_file workspace_dir ci_commit_sha"
        )
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    workspace_dir = sys.argv[3]
    ci_commit_sha = sys.argv[4]

    yaml_out = parse_procedure(input_file, workspace_dir, ci_commit_sha)

    with Path(output_file).open(mode="w", encoding="utf-8") as f:
        yaml.dump(yaml_out, f, Dumper=MyDumper, default_flow_style=False)
