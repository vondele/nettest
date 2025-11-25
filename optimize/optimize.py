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
        w1,
        w2,
    ):
        with self.lock:
            self.exec_id += 1
            local_exec_id = self.exec_id

        print(
            f"Starting {local_exec_id}:",
            w1,
            w2,
            flush=True,
        )

        # lr = 0.0010783148702050778 * math.pow(2.0, lr_scaling_power)
        # gamma = 1.0 - gamma_adjust / 1000.0

        recipe_str = f"""
testing:
  fastchess:
    code:
      owner: Disservin
      sha: e06cb1a55d80d768fdce29771b1749b40daeeab3
    options:
      hash: 16
      max_rounds: 80000
      tc: 10+0.1
    sprt:
      nElo_interval_midpoint: {self.nElo_target}
      nElo_interval_width: 2
  reference:
    code:
      owner: vondele
      sha: 928557ead60b0a34d32c965d1ee5ca2178239b79
      target: profile-build
  steps: last
  testing:
    code:
      owner: vondele
      sha: 928557ead60b0a34d32c965d1ee5ca2178239b79
      target: profile-build
training:
  steps:
    - convert:
        binpack: official-stockfish/master-binpacks/fishpack32.binpack
        checkpoint2nnue:
          - --features=Full_Threats^
          - --l1=1024
          - --ft_compression=leb128
        optimize:
          - --features=Full_Threats
          - --l1=1024
          - --ft_optimize_count=100000
          - --ft_optimize
          - --ft_compression=leb128
      run:
        binpacks:
          - vondele/from_kaggle_2/T60T70wIsRightFarseerT60T74T75T76.split_0.binpack
          - vondele/from_kaggle_2/T60T70wIsRightFarseerT60T74T75T76.split_1.binpack
          - vondele/from_kaggle_2/T60T70wIsRightFarseerT60T74T75T76.split_2.binpack
          - vondele/from_kaggle_2/T60T70wIsRightFarseerT60T74T75T76.split_3.binpack
          - vondele/from_kaggle_2/T60T70wIsRightFarseerT60T74T75T76.split_4.binpack
          - official-stockfish/master-binpacks/nodes5000pv2_UHO.binpack
          - official-stockfish/master-binpacks/wrongIsRight_nodes5000pv2.binpack
          - official-stockfish/master-binpacks/multinet_pv-2_diff-100_nodes-5000.binpack
          - official-stockfish/master-binpacks/dfrc_n5000.binpack
        max_epochs: 800
        other_options:
          - --batch-size=65536
          - --features=Full_Threats^
          - --l1=1024
          - --lr=4.375e-4
          - --gamma=0.995
          - --start-lambda=1.0
          - --end-lambda=0.75
          - --random-fen-skipping=10
          - --early-fen-skipping=12
        repetitions: 3
        resume: none
      trainer:
        owner: sscg13
        sha: 68b56ad3bfa98a6433b3e37fd4b26ba9155fbf2c
    - convert:
        binpack: official-stockfish/master-binpacks/fishpack32.binpack
        checkpoint2nnue:
          - --features=Full_Threats^
          - --l1=1024
          - --ft_compression=leb128
        optimize:
          - --features=Full_Threats
          - --l1=1024
          - --ft_optimize_count=100000
          - --ft_optimize
          - --ft_compression=leb128
      run:
        binpacks:
          - vondele/from_kaggle_1/leela96-filt-v2.min.split_0.binpack
          - vondele/from_kaggle_1/leela96-filt-v2.min.split_1.binpack
          - vondele/from_kaggle_1/leela96-filt-v2.min.split_2.binpack
          - vondele/from_kaggle_1/leela96-filt-v2.min.split_3.binpack
          - vondele/from_kaggle_1/leela96-filt-v2.min.split_4.binpack
          - linrock/test60/test60-2021-11-nov-12tb7p.min-v2.binpack
          - linrock/test60/test60-2021-12-dec-12tb7p.min-v2.binpack
          - linrock/test78/test78-2022-01-to-05-jantomay-16tb7p.v6-dd.min.binpack
          - linrock/test78/test78-2022-06-to-09-juntosep-16tb7p.v6-dd.min.binpack
          - linrock/test77/test77-2021-12-dec-16tb7p.v6-dd.min.binpack
          - linrock/test78/test78-2022-01-to-05-jantomay-16tb7p.v6-dd.min.binpack
          - linrock/test78/test78-2022-06-to-09-juntosep-16tb7p.v6-dd.min.binpack
          - linrock/test78/test78-2022-01-to-05-jantomay-16tb7p.v6-dd.min.binpack
          - linrock/test78/test78-2022-06-to-09-juntosep-16tb7p.v6-dd.min.binpack
          - linrock/test79/test79-2022-04-apr-16tb7p.v6-dd.min.binpack
          - linrock/test79/test79-2022-05-may-16tb7p.v6-dd.min.binpack
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
        max_epochs: 800
        other_options:
          - --batch-size=65536
          - --features=Full_Threats^
          - --l1=1024
          - --lr=0.0010783148702050778
          - --gamma=0.994446435229841
          - --pow-exp=2.442037790427722
          - --qp-asymmetry=0.15795949371005746
          - --start-lambda=0.8460635942002347
          - --end-lambda=0.7490913625693039
          - --in-scaling=294.7193839807687
          - --out-scaling=364.552099066772
          - --in-offset=281.4186220835457
          - --out-offset=279.93991915496105
          - --random-fen-skipping=10
          - --early-fen-skipping=28
        repetitions: 3
        resume: previous_model
      trainer:
        owner: sscg13
        sha: 68b56ad3bfa98a6433b3e37fd4b26ba9155fbf2c
    - convert:
        binpack: official-stockfish/master-binpacks/fishpack32.binpack
        checkpoint2nnue:
          - --features=Full_Threats^
          - --l1=1024
          - --ft_compression=leb128
        optimize:
          - --features=Full_Threats
          - --l1=1024
          - --ft_optimize_count=100000
          - --ft_optimize
          - --ft_compression=leb128
      run:
        binpacks:
          - vondele/from_kaggle_1/leela96-filt-v2.min.split_0.binpack
          - vondele/from_kaggle_1/leela96-filt-v2.min.split_1.binpack
          - vondele/from_kaggle_1/leela96-filt-v2.min.split_2.binpack
          - vondele/from_kaggle_1/leela96-filt-v2.min.split_3.binpack
          - vondele/from_kaggle_1/leela96-filt-v2.min.split_4.binpack
          - linrock/test60/test60-2021-11-nov-12tb7p.min-v2.binpack
          - linrock/test60/test60-2021-12-dec-12tb7p.min-v2.binpack
          - linrock/test78/test78-2022-01-to-05-jantomay-16tb7p.v6-dd.min.binpack
          - linrock/test78/test78-2022-06-to-09-juntosep-16tb7p.v6-dd.min.binpack
          - linrock/test77/test77-2021-12-dec-16tb7p.v6-dd.min.binpack
          - linrock/test78/test78-2022-01-to-05-jantomay-16tb7p.v6-dd.min.binpack
          - linrock/test78/test78-2022-06-to-09-juntosep-16tb7p.v6-dd.min.binpack
          - linrock/test78/test78-2022-01-to-05-jantomay-16tb7p.v6-dd.min.binpack
          - linrock/test78/test78-2022-06-to-09-juntosep-16tb7p.v6-dd.min.binpack
          - linrock/test79/test79-2022-04-apr-16tb7p.v6-dd.min.binpack
          - linrock/test79/test79-2022-05-may-16tb7p.v6-dd.min.binpack
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
        max_epochs: 800
        other_options:
          - --batch-size=65536
          - --features=Full_Threats^
          - --l1=1024
          - --lr=0.0010783148702050778
          - --gamma=0.994446435229841
          - --pow-exp=2.442037790427722
          - --qp-asymmetry=0.15795949371005746
          - --start-lambda=0.8460635942002347
          - --end-lambda=0.7490913625693039
          - --in-scaling=294.7193839807687
          - --out-scaling=364.552099066772
          - --in-offset=281.4186220835457
          - --out-offset=279.93991915496105
          - --random-fen-skipping=10
          - --early-fen-skipping=28
        repetitions: 3
        resume: previous_model
      trainer:
        owner: sscg13
        sha: 68b56ad3bfa98a6433b3e37fd4b26ba9155fbf2c
    - convert:
        binpack: official-stockfish/master-binpacks/fishpack32.binpack
        checkpoint2nnue:
          - --features=Full_Threats^
          - --l1=1024
          - --ft_compression=leb128
        optimize:
          - --features=Full_Threats
          - --l1=1024
          - --ft_optimize_count=100000
          - --ft_optimize
          - --ft_compression=leb128
      run:
        binpacks:
          - vondele/from_kaggle_1/leela96-filt-v2.min.split_0.binpack
          - vondele/from_kaggle_1/leela96-filt-v2.min.split_1.binpack
          - vondele/from_kaggle_1/leela96-filt-v2.min.split_2.binpack
          - vondele/from_kaggle_1/leela96-filt-v2.min.split_3.binpack
          - vondele/from_kaggle_1/leela96-filt-v2.min.split_4.binpack
          - linrock/test60/test60-2021-11-nov-12tb7p.min-v2.binpack
          - linrock/test60/test60-2021-12-dec-12tb7p.min-v2.binpack
          - linrock/test78/test78-2022-01-to-05-jantomay-16tb7p.v6-dd.min.binpack
          - linrock/test78/test78-2022-06-to-09-juntosep-16tb7p.v6-dd.min.binpack
          - linrock/test77/test77-2021-12-dec-16tb7p.v6-dd.min.binpack
          - linrock/test78/test78-2022-01-to-05-jantomay-16tb7p.v6-dd.min.binpack
          - linrock/test78/test78-2022-06-to-09-juntosep-16tb7p.v6-dd.min.binpack
          - linrock/test78/test78-2022-01-to-05-jantomay-16tb7p.v6-dd.min.binpack
          - linrock/test78/test78-2022-06-to-09-juntosep-16tb7p.v6-dd.min.binpack
          - linrock/test79/test79-2022-04-apr-16tb7p.v6-dd.min.binpack
          - linrock/test79/test79-2022-05-may-16tb7p.v6-dd.min.binpack
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
        max_epochs: 800
        other_options:
          - --batch-size=65536
          - --features=Full_Threats^
          - --l1=1024
          - --lr=0.0010783148702050778
          - --gamma=0.994446435229841
          - --pow-exp=2.442037790427722
          - --qp-asymmetry=0.15795949371005746
          - --start-lambda=0.8460635942002347
          - --end-lambda=0.7490913625693039
          - --in-scaling=294.7193839807687
          - --out-scaling=364.552099066772
          - --in-offset=281.4186220835457
          - --out-offset=279.93991915496105
          - --random-fen-skipping=10
          - --early-fen-skipping=28
        repetitions: 3
        resume: previous_model
      trainer:
        owner: sscg13
        sha: 68b56ad3bfa98a6433b3e37fd4b26ba9155fbf2c
    - convert:
        binpack: official-stockfish/master-binpacks/fishpack32.binpack
        checkpoint2nnue:
          - --features=Full_Threats^
          - --l1=1024
          - --ft_compression=leb128
        optimize:
          - --features=Full_Threats
          - --l1=1024
          - --ft_optimize_count=100000
          - --ft_optimize
          - --ft_compression=leb128
      run:
        binpacks:
          - vondele/from_kaggle_1/leela96-filt-v2.min.split_0.binpack
          - vondele/from_kaggle_1/leela96-filt-v2.min.split_1.binpack
          - vondele/from_kaggle_1/leela96-filt-v2.min.split_2.binpack
          - vondele/from_kaggle_1/leela96-filt-v2.min.split_3.binpack
          - vondele/from_kaggle_1/leela96-filt-v2.min.split_4.binpack
          - linrock/test60/test60-2021-11-nov-12tb7p.min-v2.binpack
          - linrock/test60/test60-2021-12-dec-12tb7p.min-v2.binpack
          - linrock/test77/test77-2021-12-dec-16tb7p.v6-dd.min.binpack
          - linrock/test78/test78-2022-01-to-05-jantomay-16tb7p.v6-dd.min.binpack
          - linrock/test78/test78-2022-06-to-09-juntosep-16tb7p.v6-dd.min.binpack
          - linrock/test79/test79-2022-04-apr-16tb7p.v6-dd.min.binpack
          - linrock/test79/test79-2022-05-may-16tb7p.v6-dd.min.binpack
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
        max_epochs: 800
        other_options:
          - --batch-size=65536
          - --features=Full_Threats^
          - --l1=1024
          - --lr=0.00042301976890599417
          - --gamma=0.9935974858411222
          - --pow-exp=2.442037790427722
          - --qp-asymmetry=0.15795949371005746
          - --start-lambda=0.8460635942002347
          - --end-lambda=0.7404368679861961
          - --in-scaling=294.7193839807687
          - --out-scaling=352.8750799744594
          - --in-offset=281.4186220835457
          - --out-offset=279.93991915496105
          - --random-fen-skipping=10
          - --early-fen-skipping=28
          - --pc-y1=0.6893201149773951
          - --pc-y2=2.9285769485515805
          - --pc-y3=1.4386005301749225
          - --w1={w1}
          - --w2={w2}
        repetitions: 3
        resume: previous_model
      trainer:
        owner: vondele
        sha: 81b8c11c2f673a88fc4dff516ec655d3daf7124c
        """
        recipe = yaml.safe_load(recipe_str)

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
            w1,
            w2,
            nElo,
            bestNet,
            flush=True,
        )

        if nElo > self.nElo_target:
            self.nElo_target = min(nElo, self.nElo_target + 1)  # could also be nElo

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
        environment = dict()

    instrumentation = ng.p.Instrumentation(
        # ng.p.Scalar(init=28)
        # .set_bounds(lower=10, upper=32)
        # .set_mutation(sigma=3.0)
        # .set_integer_casting(),  # early_fen_skipping
        ng.p.Scalar(init=0.0)
        .set_bounds(lower=-10, upper=10)
        .set_mutation(sigma=1.0),  # w1
        ng.p.Scalar(init=0.5)
        .set_bounds(lower=0.1, upper=1.0)
        .set_mutation(sigma=0.1),  # w2
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
        #        ng.p.Scalar(init=2.5)  # pow_exp
        #        .set_bounds(lower=2.0, upper=3.0)
        #        .set_mutation(sigma=0.1),
        #        ng.p.Scalar(init=0.15)  # qp_asymmetry
        #        .set_bounds(lower=0.0, upper=0.3)
        #        .set_mutation(sigma=0.03),
        #        ng.p.Scalar(init=0.8)  # start_lambda
        #        .set_bounds(lower=0.7, upper=0.9)
        #        .set_mutation(sigma=0.02),
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

    budget = 64  # Total number of evaluations to perform
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
