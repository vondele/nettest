#!/bin/bash

set -e

cat <<EOF
# A first step of training. Probably doing multiple step can be organized a bit differently, possibly dynamically ? TODO: --ft_optimize (too slow for testing)
training-run1:
  timeout: 48h
  stage: runTraining
  script:
    - cd /workspace/nnue-pytorch
    - export DATASETS="/workspace/data/from_classical_04_pv-2_diff-100_nodes-5000.binpack"
    - export ROOTDIR=/workspace/scratch/\$CI_COMMIT_SHA/training/runs/run_1
    - export RESUMEOPT=""
    - python train.py \$DATASETS --gpus="0," --threads 16 --num-workers 16 --max_epochs 2 --network-save-period 2 --enable_progress_bar false  --batch-size 16384 --random-fen-skipping 3 --features=HalfKAv2_hm^ --default_root_dir \$ROOTDIR \$RESUMEOPT
    - export OMP_NUM_THREADS=8
    - export LASTCKPT=\$ROOTDIR/lightning_logs/version_0/checkpoints/last.ckpt
    - export LASTNNUE=\$ROOTDIR/lightning_logs/version_0/checkpoints/last.nnue
    - export FTOPTDATA=/workspace/data/official-stockfish/master-binpacks/test80-2022-08-aug-16tb7p.v6-dd.min.binpack
    - python serialize.py \$LASTCKPT \$LASTNNUE --features=HalfKAv2_hm^ --ft_compression=leb128 --ft_optimize_count=1000000 --ft_optimize_data=\$FTOPTDATA
    - export NNUESHA=\$(sha256sum \$LASTNNUE | cut -c1-12)
    - echo "copying \$LASTNNUE to nn-\$NNUESHA.nnue"
    - mkdir -p \$CI_PROJECT_DIR/networks/
    - cp \$LASTNNUE \$CI_PROJECT_DIR/networks/nn-\$NNUESHA.nnue
  artifacts:
    expire_in: 1 week
    paths:
      - networks

# A second step of training. Not it resumes and continues a previous trainig. Probably doing multiple step can be organized a bit differently, possibly dynamically ? TODO: --ft_optimize (too slow for testing)
training-run2:
  timeout: 48h
  stage: runTraining
  needs: [training-run1]
  script:
    - cd /workspace/nnue-pytorch
    - export DATASETS="/workspace/data/from_classical_04_pv-2_diff-100_nodes-5000.binpack"
    - export ROOTDIR=/workspace/scratch/\$CI_COMMIT_SHA/training/runs/run_2
    - export RESUMEOPT="--resume_from_checkpoint /workspace/scratch/\$CI_COMMIT_SHA/training/runs/run_1/lightning_logs/version_0/checkpoints/last.ckpt"
    - python train.py \$DATASETS --gpus="0," --threads 16 --num-workers 16 --max_epochs 4 --network-save-period 2 --enable_progress_bar false  --batch-size 16384 --random-fen-skipping 3 --features=HalfKAv2_hm^ --default_root_dir \$ROOTDIR \$RESUMEOPT
    - export OMP_NUM_THREADS=8
    - export LASTCKPT=\$ROOTDIR/lightning_logs/version_0/checkpoints/last.ckpt
    - export LASTNNUE=\$ROOTDIR/lightning_logs/version_0/checkpoints/last.nnue
    - export FTOPTDATA=/workspace/data/official-stockfish/master-binpacks/test80-2022-08-aug-16tb7p.v6-dd.min.binpack
    - python serialize.py \$LASTCKPT \$LASTNNUE --features=HalfKAv2_hm^ --ft_compression=leb128 --ft_optimize_count=1000000 --ft_optimize_data=\$FTOPTDATA
    - export NNUESHA=\$(sha256sum \$LASTNNUE | cut -c1-12)
    - echo "copying \$LASTNNUE to nn-\$NNUESHA.nnue"
    - mkdir -p \$CI_PROJECT_DIR/networks/
    - cp \$LASTNNUE \$CI_PROJECT_DIR/networks/nn-\$NNUESHA.nnue
  artifacts:
    expire_in: 1 week
    paths:
      - networks
EOF


