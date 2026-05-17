from firecrest_executor import FirecrestExecutor
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from nettest import execute
import nevergrad as ng
import yaml
import math
import argparse
import threading
from pathlib import Path
from nettest.utils import MyDumper
from nettest.default_environment import get_default_environment


class RemoteNet:
    def __init__(self, environment, max_workers=16, local=False, nElo_target=0):
        # we could probably share the executor between calls to train_and_test_net
        if local:
            self.executor = ProcessPoolExecutor(max_workers=max_workers)
        else:
            self.executor = FirecrestExecutor(
                working_dir="/users/vjoost/fish/workspace/",
                sbatch_options=[
                    "--job-name=FirecrestExecutor",
                    "--time=12:00:00",
                    "--nodes=1",
                    "--partition=normal",
                ],
                srun_options=[
                    "--environment=/users/vjoost/fish/workspace/nettest.toml"
                ],
                sleep_interval=10,
                max_workers=max_workers,
                task_retries=30,
            )
        self.environment = environment
        self.nElo_target = nElo_target
        self.lock = threading.Lock()
        self.exec_id = 0

    # the recipe to optimize
    def train_and_test_net(
        self,
        pow_exp,
        qp_asym,
        lambda_sample,
        lambda_batch,
        both_lambda,
    ):
        with self.lock:
            self.exec_id += 1
            local_exec_id = self.exec_id

        print(
            f"Starting {local_exec_id}:",
            pow_exp,
            qp_asym,
            lambda_sample,
            lambda_batch,
            both_lambda,
            flush=True,
        )

        recipe_str = f"""
stage_one_binpacks: &stage_one_binpacks
  - vondele/from_kaggle_2/T60T70wIsRightFarseerT60T74T75T76.split_0.binpack
  - vondele/from_kaggle_2/T60T70wIsRightFarseerT60T74T75T76.split_1.binpack
  - vondele/from_kaggle_2/T60T70wIsRightFarseerT60T74T75T76.split_2.binpack
  - vondele/from_kaggle_2/T60T70wIsRightFarseerT60T74T75T76.split_3.binpack
  - vondele/from_kaggle_2/T60T70wIsRightFarseerT60T74T75T76.split_4.binpack
  - official-stockfish/master-binpacks/nodes5000pv2_UHO.binpack
  - official-stockfish/master-binpacks/wrongIsRight_nodes5000pv2.binpack
  - official-stockfish/master-binpacks/multinet_pv-2_diff-100_nodes-5000.binpack
  - official-stockfish/master-binpacks/dfrc_n5000.binpack

stage_two_plus_binpacks: &stage_two_plus_binpacks
  - vondele/from_kaggle_1/leela96-filt-v2.min.split_0.binpack
  - vondele/from_kaggle_1/leela96-filt-v2.min.split_1.binpack
  - vondele/from_kaggle_1/leela96-filt-v2.min.split_2.binpack
  - vondele/from_kaggle_1/leela96-filt-v2.min.split_3.binpack
  - vondele/from_kaggle_1/leela96-filt-v2.min.split_4.binpack
  - linrock/test60/test60-2021-11-nov-12tb7p.min-v2.binpack
  - linrock/test60/test60-2021-12-dec-12tb7p.min-v2.binpack
  - linrock/test77/test77-2021-12-dec-16tb7p.v6-dd.min.binpack
  - linrock/test78/test78-2022-01-to-05-jantomay-16tb7p.v6-dd.min.binpack
  - linrock/test79/test79-2022-04-apr-16tb7p.v6-dd.min.binpack
  - linrock/test79/test79-2022-05-may-16tb7p.v6-dd.min.binpack
  - linrock/test78/test78-2022-06-to-09-juntosep-16tb7p.v6-dd.min.binpack
  - linrock/test80-2022/test80-2022-06-jun-16tb7p.v6-dd.min.binpack
  - linrock/test80-2022/test80-2022-07-jul-16tb7p.v6-dd.min.binpack
  - linrock/test80-2022/test80-2022-08-aug-16tb7p.v6-dd.min.binpack
  - linrock/test80-2022/test80-2022-09-sep-16tb7p.v6-dd.min.binpack
  - linrock/test80-2022/test80-2022-10-oct-16tb7p.v6-dd.binpack
  - linrock/test80-2022/test80-2022-11-nov-16tb7p.v6-dd.min.binpack
  - linrock/test80-2023/test80-2023-01-jan-16tb7p.v6-sk20.min.binpack
  - linrock/test80-2023/test80-2023-02-feb-16tb7p.v6-dd.min.binpack
  - linrock/test80-2023/test80-2023-03-mar-2tb7p.v6-sk16.min.binpack
  - linrock/test80-2023/test80-2023-04-apr-2tb7p.v6-sk16.min.binpack
  - linrock/test80-2023/test80-2023-05-may-2tb7p.v6.min.binpack
  - linrock/test80-2023/test80-2023-06-jun-2tb7p.min-v2.v6.binpack
  - linrock/test80-2023/test80-2023-07-jul-2tb7p.min-v2.v6.binpack
  - linrock/test80-2023/test80-2023-08-aug-2tb7p.v6.min.binpack
  - linrock/test80-2023/test80-2023-09-sep-2tb7p.min-v2.v6.binpack
  - linrock/test80-2023/test80-2023-10-oct-2tb7p.min-v2.v6.binpack
  - linrock/test80-2023/test80-2023-11-nov-2tb7p.min-v2.v6.binpack
  - linrock/test80-2023/test80-2023-12-dec-2tb7p.min-v2.v6.binpack
  - linrock/test80-2024/test80-2024-01-jan-2tb7p.min-v2.v6.binpack
  - linrock/test80-2024/test80-2024-02-feb-2tb7p.min-v2.v6.binpack

optimize_flags: &optimize_flags
  features: Full_Threats+HalfKAv2_hm^
  l1: 1024
  l2: 31
  ft-optimize-count: 100000
  ft-optimize: true
  ft-compression: leb128

checkpoint2nnue_options: &checkpoint2nnue_options
  features: Full_Threats+HalfKAv2_hm^
  l1: 1024
  l2: 31
  ft-compression: leb128

common_run_options: &common_run_options
  optimizer-name: rangerlite
  validation-size: 250000000
  check-val-every-n-epoch: 25
  batch-size: 65536
  features: Full_Threats+HalfKAv2_hm^
  l1: 1024
  l2: 31
  random-fen-skipping: 10
  early-fen-skipping: 18
  soft-early-fen_skipping: 32
  pc-y0: -0.20
  pc-y1: 0.45
  pc-y2: 1.0
  pc-y3: 0.95
  pc-y4: 0.75

advanced_stage_options: &advanced_stage_options
  ply-x1: 0.00
  ply-y1: 0.025
  ply-x2: 22.0
  ply-y2: 0.05
  ply-x3: 25.5
  ply-y3: 0.20
  ply-x4: 29.5
  ply-y4: 0.80
  pow-exp: {pow_exp}
  qp-asymmetry: {qp_asym}
  in-scaling: 294.7193839807687
  out-scaling: 352.8750799744594
  in-offset: 281.4186220835457
  out-offset: 279.93991915496105
  one-cycle-warmup-pct: 0.05
  one-cycle-final-div: "1e3"
  jitter-lambda-sample: {lambda_sample}
  jitter-lambda-batch: {lambda_batch}
  jitter-decay-lambda-batch: 0.999
  start-lambda: {both_lambda}
  end-lambda: {both_lambda}

trainer: &trainer
  owner: official-stockfish
  sha: 77038253250ee9d672e29cc9fb75ab4971e09ce0

# --- Grouped Structure Anchors for Redundancy Reduction ---
common_convert: &common_convert
  binpack: official-stockfish/master-binpacks/fishpack32.binpack
  checkpoint2nnue: *checkpoint2nnue_options
  optimize: *optimize_flags

common_step: &common_step
  convert: *common_convert
  trainer: *trainer

common_run_stage_two_plus: &common_run_stage_two_plus
  binpacks: *stage_two_plus_binpacks
  repetitions: 1
  resume: previous_model

common_run_checkpoint: &common_run_checkpoint
  <<: *common_run_stage_two_plus
  resume: previous_checkpoint

official_code: &official_code
  owner: official-stockfish
  sha: dd321af5dfc0789de07c4e5c64915073995eb818
  target: profile-build
# ----------------------------------------------------------

testing:
  crosscheck:
    trainer: *trainer
    binpack: official-stockfish/master-binpacks/fishpack32.binpack
    other_options:
      features: Full_Threats+HalfKAv2_hm^
      l1: 1024
      l2: 31
      count: 5000
  fastchess:
    code:
      owner: Disservin
      sha: 541aef889206882de4823cf5a676d53ac5171c6f
    options:
      hash: 16
      max_rounds: 60000
      tc: 10+0.1
  reference:
    code: *official_code
  steps: last
  testing:
    code: *official_code

training:
  steps:
    # Stage 1
    - <<: *common_step
      run:
        binpacks: *stage_one_binpacks
        max_epochs: 250
        other_options:
          <<: *common_run_options
          one-cycle-steps: 381500
          lr: "2e-3"
          start-lambda: 1.0
          end-lambda: 0.75
          ply-x1: 0.0
          ply-y1: 0.01
          ply-x2: 14.0
          ply-y2: 0.20
          ply-x3: 18.5
          ply-y3: 0.50
          ply-x4: 29.5
          ply-y4: 0.80
        repetitions: 1
        resume: none

    # Stage 2
    - <<: *common_step
      run:
        <<: *common_run_stage_two_plus
        max_epochs: 300
        other_options:
          <<: [*common_run_options, *advanced_stage_options]
          one-cycle-steps: 381500
          lr: "1.5e-3"
          swa-start-epoch: 250

    # Stage 3
    - <<: *common_step
      run:
        <<: *common_run_stage_two_plus
        max_epochs: 550
        other_options:
          <<: [*common_run_options, *advanced_stage_options]
          one-cycle-steps: 763000
          lr: "1e-3"
          swa-start-epoch: 500

    # Stage 4
    - <<: *common_step
      run:
        <<: *common_run_stage_two_plus
        max_epochs: 1050
        other_options:
          <<: [*common_run_options, *advanced_stage_options]
          one-cycle-steps: 1526000
          lr: "0.67e-3"
          swa-start-epoch: 1000

    # Stage 5
    - <<: *common_step
      run:
        <<: *common_run_stage_two_plus
        max_epochs: 1000
        other_options:
          <<: [*common_run_options, *advanced_stage_options]
          one-cycle-steps: 3052000
          lr: "0.5e-3"
          swa-start-epoch: 2000

    # Stage 6
    - <<: *common_step
      run:
        <<: *common_run_checkpoint
        max_epochs: 1500
        other_options:
          <<: [*common_run_options, *advanced_stage_options]
          one-cycle-steps: 3052000
          lr: "0.5e-3"
          swa-start-epoch: 2000

    # Stage 7
    - <<: *common_step
      run:
        <<: *common_run_checkpoint
        max_epochs: 2050
        other_options:
          <<: [*common_run_options, *advanced_stage_options]
          one-cycle-steps: 3052000
          lr: "0.5e-3"
          swa-start-epoch: 2000
        """
        recipe = yaml.safe_load(recipe_str)

        with Path(f"optimize_recipe_str_{local_exec_id}.yaml").open(
            mode="w", encoding="utf-8"
        ) as f:
            f.write(recipe_str)

        with Path(f"optimize_recipe_{local_exec_id}.yaml").open(
            mode="w", encoding="utf-8"
        ) as f:
            yaml.dump(recipe, f, Dumper=MyDumper, default_flow_style=False, width=300)

        bestNet, nElo = execute(
            recipe=recipe,
            executor=self.executor,
            environment=self.environment,
        )

        if nElo is None:
            # TODO ... better error handling possible ?
            print(
                "Something went wrong during evaluation .... continuing with lower bound estimate"
            )
            nElo = self.nElo_target - 10

        print(
            f"Done {local_exec_id}:",
            pow_exp,
            qp_asym,
            lambda_sample,
            lambda_batch,
            both_lambda,
            nElo,
            bestNet,
            flush=True,
        )

        if nElo > self.nElo_target:
            self.nElo_target = self.nElo_target + 1  # could also be nElo

        return -nElo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize a recipe")
    parser.add_argument(
        "--environment", required=False, help="Definition of the environment file"
    )
    args = parser.parse_args()

    if args.environment:
        print("Using environment file: ", args.environment)
        with open(args.environment) as f:
            environment = yaml.safe_load(f)
    else:
        environment = get_default_environment()

    instrumentation = ng.p.Instrumentation(
        # ng.p.Scalar(init=28)
        # .set_bounds(lower=10, upper=32)
        # .set_mutation(sigma=3.0)
        # .set_integer_casting(),  # early_fen_skipping
        # ng.p.Scalar(init=1.0)
        # .set_bounds(lower=0.5, upper=2.0)
        # .set_mutation(sigma=0.2),  # pc_y1
        # ng.p.Scalar(init=2.0)
        # .set_bounds(lower=1.0, upper=4.0)
        # .set_mutation(sigma=0.5),  # pc_y2
        # ng.p.Scalar(init=1.0)
        # .set_bounds(lower=0.5, upper=2.0)
        # .set_mutation(sigma=0.2),  # pc_y3
        # ng.p.Scalar(init=0.0)
        # .set_bounds(lower=-4.0, upper=4.0)
        # .set_mutation(sigma=1.0),  # lr_scaling_power
        # ng.p.Scalar(init=5.0)
        # .set_bounds(lower=0.0, upper=10.0)
        # .set_mutation(sigma=1.0),  # gamma_adjust
        ng.p.Scalar(init=2.442037790427722)  # pow_exp
        .set_bounds(lower=2.0, upper=3.0)
        .set_mutation(sigma=0.03),
        ng.p.Scalar(init=0.15795949371005746)  # qp_asymmetry
        .set_bounds(lower=0.0, upper=0.3)
        .set_mutation(sigma=0.03),
        ng.p.Scalar(init=0.0030)  # lambda_sample
        .set_bounds(lower=0.0000, upper=0.0090)
        .set_mutation(sigma=0.001),
        ng.p.Scalar(init=0.0100)  # lambda_batch
        .set_bounds(lower=0.0000, upper=0.0300)
        .set_mutation(sigma=0.003),
        ng.p.Scalar(init=0.76)  # both_lambda
        .set_bounds(lower=0.6, upper=0.9)
        .set_mutation(sigma=0.02),
        # ng.p.Scalar(init=0.75)  # end_lambda
        # .set_bounds(lower=0.55, upper=0.85)
        # .set_mutation(sigma=0.04),
        # ng.p.Scalar(init=297.1)  # in_scaling
        # .set_bounds(lower=214, upper=374)
        # .set_mutation(sigma=10),
        # ng.p.Scalar(init=361)  # out_scaling
        # .set_bounds(lower=284, upper=444)
        # .set_mutation(sigma=10),
        #        ng.p.Scalar(init=270)  # in_offset
        #        .set_bounds(lower=210, upper=330)
        #        .set_mutation(sigma=10),
        #        ng.p.Scalar(init=270)  # out_offset
        #        .set_bounds(lower=210, upper=330)
        #        .set_mutation(sigma=10),
    )

    budget = 128  # Total number of evaluations to perform
    num_workers = 16  # Number of parallel workers to use

    # The remotely trainable net
    remoteNet = RemoteNet(
        environment=environment, max_workers=num_workers, local=False, nElo_target=1
    )

    # Use TBPSA optimizer
    optimizer = ng.optimizers.TBPSA(
        instrumentation, budget=budget, num_workers=num_workers
    )

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        recommendation = optimizer.minimize(
            remoteNet.train_and_test_net, executor=executor
        )

    print("Final Recommended solution:", recommendation.value)
