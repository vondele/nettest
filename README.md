# Nettest

This repo aims to capture the workflow needed to train a NNUE network in a
single yaml file. It contains these yaml files (recipes) and the tools to
execute the workflow in a reproducible manner. Ultimately the goal is to
reproduce the workflows needed to create (near?) master-strength networks for
[Stockfish](https://github.com/official-stockfish/Stockfish).

## Overview

Currently, the repo contains two main recipes:

* [small.yaml](small.yaml): generating the small net in stockfish
* [large.yaml](large.yaml): generating the large/big net in stockfish

and two auxiliary ones for testing and experimentation.

There are currently three ways to execute such a pipeline:

* Through a local containerized setup: for most users.
* Through a gitlab CI pipeline: original intended use.
* Through a remote containerized setup: advanced, requires system access.

The idea is that new and improved recipes can be generated and tested locally,
but that new net recipes are merged as they pass the CI pipeline.

In principle, users should not need to modify anything except the recipes.
The main tools and scripts are in the `nettest` directory, the `ci` directory
contains configurations needed for the CI pipeline, including the container
definition, and `optimize` contains scripts to optimize hyperparameters.

## Execution of a recipe

### General

Execution of a recipe includes downloading data from huggingface, training the
network in various stages, with restarts, exporting and optimizing the networks
for inference, and testing the resulting network. All these steps are encoded
in the yaml, and executed automatically.

The storage and compute needs are significant. For training the large net,
approximately 700GB of storage is needed, training both small and large nets
requires 900GB. On a high-end GPU, training a large net takes approximately 4-5
days, whereas a small net takes less than one day. 8GB GPU RAM is needed to
train small nets more than that for the large nets. Testing the resulting nets
will take multiple hours, even at high CPU concurrency.

The workflows have built-in caching mechanisms, for both data and compute. In
particular, data needed for training is downloaded once and cached, and
identical training steps that have previously completed successfully in other
workflows are not repeated. This allows for quicker iteration when modifying
later steps in the workflow.

### Execution in the CI environment

An authorized person can trigger a CI pipeline for the corresponding recipe
(e.g. `small.yaml`) by simply commenting on the PR:

```txt
 cscs-ci run RECIPE=small
```

The CI pipeline will then execute the workflow, and report success or failure,
provide logs that give info on the Elo reached, as well as expose NNUE networks
as downloadable artifacts.

The workflow is generated as a dynamic pipeline using a [gitlab-ci
setup](ci/cscs.yml) that builds the container, generates the pipeline, and
executes it. The CI service is described in these [docs](https://docs.cscs.ch/services/cicd/).

### Execution in a container environment

The workflows are executed in a [container](ci/docker/Dockerfile.build) with
proper mounts. In particular, the container contains this repo as
`/workspace/nettest/`, and mounted directories `/workspace/data/`,
`/workspace/scratch/`, and `/workspace/cidir/`. The former will be used to store
and cache data downloaded from huggingface data repositories (and must offer
fast access, e.g. SSD), while the latter contain training and testing run
intermediate data, checkpoints and nets.

#### local execution

Assuming a working docker setup (exposing the GPU of the host), the following
is thus all what is needed to build the docker container and run a recipe
locally:

```bash
# clone the repo
git clone https://github.com/vondele/nettest.git
# build the container
cd nettest
docker build -t nettest.docker -f ci/docker/Dockerfile.build .
# local execution, adjust data, scratch and cidir mounts as needed
docker run -it --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
       --gpus all --cap-add=sys_nice \
       -v /mnt/ssd/data/:/workspace/data \
       -v /mnt/ssd/scratch/:/workspace/scratch \
       -v /mnt/ssd/cidir/:/workspace/cidir \
       nettest.docker /bin/bash -c \
       "python -m nettest.execute_recipe --executor local --recipe nettest/testing.yaml"
```

It is also possible to mount a local directory over `/workspace/nettest/` to be
able to easily modify and test recipes.

For more advanced hardware environments (e.g. multi GPUs, multi socket), it is
possible to influence the resource allocation by adding the argument
`--environment nettest/environments/local.yaml` (or a suitably modified yaml
file).

#### remote execution

```bash
python -m nettest.execute_recipe --executor remote --recipe nettest/large.yaml
```

Remote execution requires a proper setup, including a
local setup of the
[firecrestExecutor](https://github.com/vondele/firecrest-executor), a
description of which is beyond the scope of this document.

### Recipe description

The recipes have a training stage and a testing stage, the former potentially
consisting of multiple steps, that can be chained through various `resume`
options.

The recipes are mostly self-explanatory, except for the 'repetitions:' field,
which specifies the number of repeated attempts to run a given step. Right now,
single training runs are hard-coded to run for 12h at most, to be compatible with CI
and remote execution, and repetitions ensure that max_epochs are nevertheless
reached.

### External Tools and Data

The pipeline requires three main tools and can the recipes will specify which
variant (sha) to use:

* The [NNUE pytorch trainer](https://github.com/official-stockfish/nnue-pytorch).
* The engine [Stockfish](https://github.com/official-stockfish/Stockfish).
* The game manager [fastchess](https://github.com/Disservin/fastchess).

The training data must be made available through a huggingface repo such as,
owner and repo is inferred from the data name:

* [Official Stockfish data](https://huggingface.co/datasets/official-stockfish/)
* [Linrock's training data](https://huggingface.co/datasets/linrock/)
* [Vondele's data](https://huggingface.co/datasets/vondele/)

Auxiliary tools used in this project include:

* [nevergrad](https://github.com/FacebookResearch/Nevergrad)
* [FirecrestExecutor](https://github.com/vondele/firecrest-executor)
