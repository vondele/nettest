#!/bin/bash

cat << EOF

include:
  - remote: 'https://gitlab.com/cscs-ci/recipes/-/raw/master/templates/v2/.ci-ext.yml'

stages:
  - ensureData
  - runTrain1
  - runTrain2
  - runMatch1

# ensure data is in place
ensureData:
  timeout: 48h
  stage: ensureData
  extends: .container-runner-clariden-gh200
  image: $PERSIST_IMAGE_NAME
  script:
    - /workspace/nettest/do_ensure_data.sh /workspace/data linrock test60
    - /workspace/nettest/do_ensure_data.sh /workspace/data linrock test77
    - /workspace/nettest/do_ensure_data.sh /workspace/data linrock test78
    - /workspace/nettest/do_ensure_data.sh /workspace/data linrock test79
    - /workspace/nettest/do_ensure_data.sh /workspace/data linrock test80-2022
    - /workspace/nettest/do_ensure_data.sh /workspace/data linrock test80-2023
    - /workspace/nettest/do_ensure_data.sh /workspace/data linrock test80-2024
    - /workspace/nettest/do_ensure_data.sh /workspace/data linrock dual-nnue
    - /workspace/nettest/do_ensure_data.sh /workspace/data official-stockfish master-binpacks
    - /workspace/nettest/do_ensure_data.sh /workspace/data official-stockfish master-smallnet-binpacks
  variables:
    SLURM_JOB_NUM_NODES: 1
    SLURM_NTASKS: 1
    SLURM_TIMELIMIT: '04:00:00'

# A first step of training. Probably doing multiple step can be organized a bit differently, possibly dynamically ? 
runTrain1:
  timeout: 48h
  stage: runTrain1
  extends: .container-runner-clariden-gh200
  image: \$PERSIST_IMAGE_NAME
  script:
    - cd /workspace/nnue-pytorch
    - export DATASETS="/workspace/data/from_classical_04_pv-2_diff-100_nodes-5000.binpack"
    - export ROOTDIR=/workspace/scratch/\$CI_COMMIT_SHA/training/runs/run_1
    - export RESUMEOPT=""
    - python train.py \$DATASETS --gpus="0," --threads 16 --num-workers 16 --max_epochs 2 --network-save-period 2 --enable_progress_bar false  --batch-size 16384 --random-fen-skipping 3 --features=HalfKAv2_hm^ --default_root_dir \$ROOTDIR \$RESUMEOPT
    - /workspace/nettest/do_generate_nnue.sh \$ROOTDIR \$CI_PROJECT_DIR/networks
  variables:
    SLURM_JOB_NUM_NODES: 1
    SLURM_NTASKS: 1
    SLURM_TIMELIMIT: '12:00:00'
  artifacts:
    expire_in: 1 week
    paths:
      - networks

# A second step of training. Note it resumes and continues a previous trainig. Probably doing multiple step can be organized a bit differently, possibly dynamically ? 
runTrain2:
  timeout: 48h
  stage: runTrain2
  extends: .container-runner-clariden-gh200
  image: \$PERSIST_IMAGE_NAME
  script:
    - cd /workspace/nnue-pytorch
    - export DATASETS="/workspace/data/from_classical_04_pv-2_diff-100_nodes-5000.binpack"
    - export ROOTDIR=/workspace/scratch/\$CI_COMMIT_SHA/training/runs/run_2
    - export RESUMEOPT="--resume_from_checkpoint /workspace/scratch/\$CI_COMMIT_SHA/training/runs/run_1/lightning_logs/version_0/checkpoints/last.ckpt"
    - python train.py \$DATASETS --gpus="0," --threads 16 --num-workers 16 --max_epochs 4 --network-save-period 2 --enable_progress_bar false  --batch-size 16384 --random-fen-skipping 3 --features=HalfKAv2_hm^ --default_root_dir \$ROOTDIR \$RESUMEOPT
    - /workspace/nettest/do_generate_nnue.sh \$ROOTDIR \$CI_PROJECT_DIR/networks
  variables:
    SLURM_JOB_NUM_NODES: 1
    SLURM_NTASKS: 1
    SLURM_TIMELIMIT: '12:00:00'
  artifacts:
    expire_in: 1 week
    paths:
      - networks

# A final test with a match, need to see if we can run with 280 concurrency, probably not, rather 1 per socket?
# Should probably become a sprt.
runMatch1:
  timeout: 48h
  stage: runMatch1
  needs:
    - job: runTrain2
      artifacts: true
  extends: .container-runner-clariden-gh200
  image: $PERSIST_IMAGE_NAME
  script:
    - ls -tlr $CI_PROJECT_DIR
    - /workspace/nettest/do_run_match.sh $CI_PROJECT_DIR/networks/ /workspace/scratch/$CI_COMMIT_SHA/match 600
  variables:
    SLURM_JOB_NUM_NODES: 1
    SLURM_NTASKS: 1
    SLURM_TIMELIMIT: '04:00:00'

EOF
