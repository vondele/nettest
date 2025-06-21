# nettest

This repo aims to capture the workflow needed to train a NNUE network as a yaml
file, and ultimately reproduce the workflows needed to create (near?)
master-strength networks for
[Stockfish](https://github.com/official-stockfish/Stockfish).

The repo itself contains a few scripts to turn such a yaml description in an
pipeline that can be executed. The main purpose is to execute it as a CI
pipeline. Ultimately, the goal is to create and merge new net recipes as they
pass the CI pipeline, with the pipeline fully documenting the reproducible
workflow needed to generate it.

## Recipes and Workflows

### Overview

Currently, the repo contains three recipes:

* [large.yaml](large.yaml): generating the large/big net in stockfish
* [small.yaml](small.yaml): generating the small net in stockfish
* [testing.yaml](testing.yaml): used for testing pipelines, and scripts

An authorized person can trigger a CI pipeline for the corresponding recipe
(e.g. `small.yaml`) by commenting on the PR:
```
 cscs-ci run default;RECIPE=small
```

The recipes have a training stage and a testing stage, the former potentially
consisting of multiple steps, that can be chained through various `resume`
options. Through a caching mechanism, CI workflows will skip executing steps
that have previously been executed, allowing quicker testing of workflows that
only modify later steps, or the testing stage.

### Setup

The workflows are executed in a [container](ci/docker/Dockerfile.build), which
contains this repo as `/workspace/nettest`, and mounted directories
`/workspace/data` and `/workspace/scratch`. The former will be used to store
and cache data downloaded from huggingface data repositories, while the latter
contains and caches training and testing runs.

The workflow is generated as a dynamic pipeline using a [gitlab-ci setup](ci/cscs.yml)
that builds the container, generates the pipeline, and executes it. Experimentally,
the [pipeline generating script](generate_pipeline.py) also generates a equivalent
shell script, which can be run locally without gitlab runners, provided an
environment similar to the container.

### Tools

The pipeline requires three main tools and can be configured to build and use variants of it:

* The [NNUE pytorch trainer](https://github.com/official-stockfish/nnue-pytorch).
* The engine [Stockfish](https://github.com/official-stockfish/Stockfish).
* The game manager [fastchess](https://github.com/Disservin/fastchess).

### Input Data

The training data must be made available through a huggingface repo such as:

* [Official Stockfish master binpacks](https://huggingface.co/datasets/official-stockfish/master-binpacks)
* [Linrock's training data](https://huggingface.co/datasets/linrock/test80-2024)

Upon execution of the workflow, such repos will be cloned and the data will be available to the trainer

### Output Data

Final .nnue networks as generated through training steps, and tested in the
testing step, are available (for limited time) as artifacts in the gitlab CI
pipeline.

### Limitations

Current limitations include the fact that individual training steps are limited
in duration, to accommodate limitations in CI job duration.
