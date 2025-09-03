from firecrest_executor import FirecrestExecutor
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from nettest import execute
import nevergrad as ng
import yaml
import math


class RemoteNet:
    def __init__(self, max_workers=16, local=False, nElo_target=0):
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
        self.nElo_target = nElo_target

    # the recipe to optimize
    def train_and_test_net(
        self,
        in_scaling,
        out_scaling,
    ):
        print(
            "Starting:",
            in_scaling,
            out_scaling,
            flush=True,
        )

        # lr = 4.375e-4 * math.pow(2.0, lr_scaling_power)
        # gamma = 1.0 - gamma_adjust / 1000.0

        recipe_str = f"""
#
# accurately measured Elo for the 6 stages in this recipe
#
#   # PLAYER                  :  RATING  ERROR    POINTS   PLAYED   (%)
#   1 master                  :     0.0   ----  816891.5  1559209    52
#   2 nn-385f5e92693f.nnue    :    -6.9    0.8  188265.0   384000    49
#   3 nn-3c0ad80612b0.nnue    :    -7.3    0.8  188050.5   384000    49
#   4 nn-987723557728.nnue    :    -8.9    1.0  124814.0   256000    49
#   5 nn-c58470727ee0.nnue    :   -12.7    0.9  135585.0   281209    48
#   6 nn-2725d01cf124.nnue    :   -34.4    1.4   57377.0   127000    45
#   7 nn-b7f6fc9de040.nnue    :   -87.2    1.4   48226.0   127000    38
#
training:
  steps:
    #
    # S1 
    #
    - convert:
        binpack: official-stockfish/master-binpacks/fishpack32.binpack
        checkpoint2nnue:
          - --features=HalfKAv2_hm^
          - --l1=3072
          - --ft_compression=leb128
        optimize:
          - --features=HalfKAv2_hm
          - --l1=3072
          - --ft_optimize_count=100000
          - --ft_optimize
          - --ft_compression=leb128
      datasets:
        - hf:
            owner: official-stockfish
            repo: master-binpacks
      run:
        binpacks:
          - official-stockfish/master-binpacks/nodes5000pv2_UHO.binpack
          - official-stockfish/master-binpacks/nodes5000pv2_UHO.binpack # 2x
          - official-stockfish/master-binpacks/wrongIsRight_nodes5000pv2.binpack
          - official-stockfish/master-binpacks/multinet_pv-2_diff-100_nodes-5000.binpack
          - official-stockfish/master-binpacks/dfrc_n5000.binpack
        max_epochs: 400
        repetitions: 1
        other_options:
          - --batch-size=16384
          - --features=HalfKAv2_hm^
          - --l1=3072
          - --lr=4.375e-4
          - --gamma=0.995
          - --start-lambda=1.0
          - --end-lambda=1.0
          - --random-fen-skipping=10
        resume: none
      trainer:
        owner: official-stockfish
        sha: ba499f2819dab17ec1784ea6522ee004ff32fd57
    #
    # S2 
    #
    - convert:
        binpack: official-stockfish/master-binpacks/fishpack32.binpack
        checkpoint2nnue:
          - --features=HalfKAv2_hm^
          - --l1=3072
          - --ft_compression=leb128
        optimize:
          - --features=HalfKAv2_hm
          - --l1=3072
          - --ft_optimize_count=100000
          - --ft_optimize
          - --ft_compression=leb128
      datasets:
        - hf:
            owner: official-stockfish
            repo: master-binpacks
        - hf:
            owner: vondele
            repo: from_kaggle_2
      run:
        binpacks:
          - vondele/from_kaggle_2/T60T70wIsRightFarseerT60T74T75T76.split_0.binpack
          - vondele/from_kaggle_2/T60T70wIsRightFarseerT60T74T75T76.split_1.binpack
          - vondele/from_kaggle_2/T60T70wIsRightFarseerT60T74T75T76.split_2.binpack
          - vondele/from_kaggle_2/T60T70wIsRightFarseerT60T74T75T76.split_3.binpack
          - vondele/from_kaggle_2/T60T70wIsRightFarseerT60T74T75T76.split_4.binpack
        max_epochs: 800
        repetitions: 3
        other_options:
          - --batch-size=16384
          - --features=HalfKAv2_hm^
          - --l1=3072
          - --lr=4.375e-4
          - --gamma=0.995
          - --start-lambda=1.0
          - --end-lambda=0.75
          - --random-fen-skipping=10
          - --early-fen-skipping=12
        resume: previous_model
      trainer:
        owner: official-stockfish
        sha: ba499f2819dab17ec1784ea6522ee004ff32fd57
    #
    # S3 
    #
    - convert:
        binpack: official-stockfish/master-binpacks/fishpack32.binpack
        checkpoint2nnue:
          - --features=HalfKAv2_hm^
          - --l1=3072
          - --ft_compression=leb128
        optimize:
          - --features=HalfKAv2_hm
          - --l1=3072
          - --ft_optimize_count=100000
          - --ft_optimize
          - --ft_compression=leb128
      datasets:
        - hf:
            owner: official-stockfish
            repo: master-binpacks
        - hf:
            owner: linrock
            repo: test60
        - hf:
            owner: linrock
            repo: test77
        - hf:
            owner: linrock
            repo: test78
        - hf:
            owner: linrock
            repo: test79
        - hf:
            owner: linrock
            repo: test80-2022
        - hf:
            owner: linrock
            repo: test80-2023
        - hf:
            owner: linrock
            repo: test80-2024
        - hf:
            owner: linrock
            repo: dual-nnue
      run:
        binpacks:
          - kaggle/leela96-filt-v2.min.binpack
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
          - linrock/test80-2023/test80-2023-02-feb-16tb7p.v6-dd.min.binpack #sk20?
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
        repetitions: 3
        other_options:
          - --batch-size=16384
          - --features=HalfKAv2_hm^
          - --l1=3072
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
        resume: previous_model
      trainer:
        owner: official-stockfish
        sha: ba499f2819dab17ec1784ea6522ee004ff32fd57
    #
    # S4 repeat S3
    #
    - convert:
        binpack: official-stockfish/master-binpacks/fishpack32.binpack
        checkpoint2nnue:
          - --features=HalfKAv2_hm^
          - --l1=3072
          - --ft_compression=leb128
        optimize:
          - --features=HalfKAv2_hm
          - --l1=3072
          - --ft_optimize_count=100000
          - --ft_optimize
          - --ft_compression=leb128
      datasets:
        - hf:
            owner: official-stockfish
            repo: master-binpacks
        - hf:
            owner: linrock
            repo: test60
        - hf:
            owner: linrock
            repo: test77
        - hf:
            owner: linrock
            repo: test78
        - hf:
            owner: linrock
            repo: test79
        - hf:
            owner: linrock
            repo: test80-2022
        - hf:
            owner: linrock
            repo: test80-2023
        - hf:
            owner: linrock
            repo: test80-2024
        - hf:
            owner: linrock
            repo: dual-nnue
      run:
        binpacks:
          - kaggle/leela96-filt-v2.min.binpack
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
          - linrock/test80-2023/test80-2023-02-feb-16tb7p.v6-dd.min.binpack #sk20?
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
        repetitions: 3
        other_options:
          - --batch-size=16384
          - --features=HalfKAv2_hm^
          - --l1=3072
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
        resume: previous_model
      trainer:
        owner: official-stockfish
        sha: ba499f2819dab17ec1784ea6522ee004ff32fd57
    #
    # S5 repeat S4
    #
    - convert:
        binpack: official-stockfish/master-binpacks/fishpack32.binpack
        checkpoint2nnue:
          - --features=HalfKAv2_hm^
          - --l1=3072
          - --ft_compression=leb128
        optimize:
          - --features=HalfKAv2_hm
          - --l1=3072
          - --ft_optimize_count=100000
          - --ft_optimize
          - --ft_compression=leb128
      datasets:
        - hf:
            owner: official-stockfish
            repo: master-binpacks
        - hf:
            owner: linrock
            repo: test60
        - hf:
            owner: linrock
            repo: test77
        - hf:
            owner: linrock
            repo: test78
        - hf:
            owner: linrock
            repo: test79
        - hf:
            owner: linrock
            repo: test80-2022
        - hf:
            owner: linrock
            repo: test80-2023
        - hf:
            owner: linrock
            repo: test80-2024
        - hf:
            owner: linrock
            repo: dual-nnue
      run:
        binpacks:
          - kaggle/leela96-filt-v2.min.binpack
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
          - linrock/test80-2023/test80-2023-02-feb-16tb7p.v6-dd.min.binpack #sk20?
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
        repetitions: 3
        other_options:
          - --batch-size=16384
          - --features=HalfKAv2_hm^
          - --l1=3072
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
        resume: previous_model
      trainer:
        owner: official-stockfish
        sha: ba499f2819dab17ec1784ea6522ee004ff32fd57
    #
    # S6 repeat S5
    #
    - convert:
        binpack: official-stockfish/master-binpacks/fishpack32.binpack
        checkpoint2nnue:
          - --features=HalfKAv2_hm^
          - --l1=3072
          - --ft_compression=leb128
        optimize:
          - --features=HalfKAv2_hm
          - --l1=3072
          - --ft_optimize_count=100000
          - --ft_optimize
          - --ft_compression=leb128
      datasets:
        - hf:
            owner: official-stockfish
            repo: master-binpacks
        - hf:
            owner: linrock
            repo: test60
        - hf:
            owner: linrock
            repo: test77
        - hf:
            owner: linrock
            repo: test78
        - hf:
            owner: linrock
            repo: test79
        - hf:
            owner: linrock
            repo: test80-2022
        - hf:
            owner: linrock
            repo: test80-2023
        - hf:
            owner: linrock
            repo: test80-2024
        - hf:
            owner: linrock
            repo: dual-nnue
      run:
        binpacks:
          - kaggle/leela96-filt-v2.min.binpack
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
          - linrock/test80-2023/test80-2023-02-feb-16tb7p.v6-dd.min.binpack #sk20?
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
        repetitions: 3
        other_options:
          - --batch-size=16384
          - --features=HalfKAv2_hm^
          - --l1=3072
          - --lr=0.0010783148702050778
          - --gamma=0.994446435229841
          - --pow-exp=2.442037790427722
          - --qp-asymmetry=0.15795949371005746
          - --start-lambda=0.8460635942002347
          - --end-lambda=0.7490913625693039
          - --in-scaling={in_scaling}
          - --out-scaling={out_scaling}
          - --in-offset=281.4186220835457
          - --out-offset=279.93991915496105
          - --random-fen-skipping=10
          - --early-fen-skipping=28
        resume: previous_model
      trainer:
        owner: official-stockfish
        sha: ba499f2819dab17ec1784ea6522ee004ff32fd57
#
# Final testing stage using fastchess
#
testing:
  reference:
    code:
      owner: official-stockfish
      sha: adfddd2c984fac5f2ac02d87575af821ec118fa8
  testing:
    code:
      owner: official-stockfish
      sha: adfddd2c984fac5f2ac02d87575af821ec118fa8
  fastchess:
    code:
      owner: Disservin
      sha: 66cac47f06dc1a09d3d1865cdbf560a7814f82ea
    options:
       tc: 10+0.1
       hash: 16
    sprt:
       nElo_interval_midpoint: -15
       nElo_interval_width: 2
       max_rounds: 100000
        """
        recipe = yaml.safe_load(recipe_str)

        nElo = execute(
            recipe=recipe,
            executor=self.executor,
        )

        if nElo is None:
            # TODO ... better error handling possible ?
            print(
                "Something went wrong during evaluation .... continuing with lower bound estimate"
            )
            nElo = self.nElo_target - 10

        print(
            "Done:",
            in_scaling,
            out_scaling,
            nElo,
            flush=True,
        )

        if nElo > self.nElo_target:
            self.nElo_target = nElo

        return -nElo


if __name__ == "__main__":
    instrumentation = ng.p.Instrumentation(
        #   ng.p.Scalar(init=0.0)
        #   .set_bounds(lower=-4.0, upper=4.0)
        #   .set_mutation(sigma=1.0),  # lr_scaling_power
        #   ng.p.Scalar(init=5.0)
        #   .set_bounds(lower=0.0, upper=10.0)
        #   .set_mutation(sigma=1.0),  # gamma_adjust
        #        ng.p.Scalar(init=2.5)  # pow_exp
        #        .set_bounds(lower=2.0, upper=3.0)
        #        .set_mutation(sigma=0.1),
        #        ng.p.Scalar(init=0.15)  # qp_asymmetry
        #        .set_bounds(lower=0.0, upper=0.3)
        #        .set_mutation(sigma=0.03),
        #        ng.p.Scalar(init=0.8)  # start_lambda
        #        .set_bounds(lower=0.7, upper=0.9)
        #        .set_mutation(sigma=0.02),
        #        ng.p.Scalar(init=0.7)  # end_lambda
        #        .set_bounds(lower=0.6, upper=0.8)
        #        .set_mutation(sigma=0.02),
        ng.p.Scalar(init=297.1)  # in_scaling
        .set_bounds(lower=214, upper=374)
        .set_mutation(sigma=10),
        ng.p.Scalar(init=361)  # out_scaling
        .set_bounds(lower=284, upper=444)
        .set_mutation(sigma=10),
        #        ng.p.Scalar(init=270)  # in_offset
        #        .set_bounds(lower=210, upper=330)
        #        .set_mutation(sigma=10),
        #        ng.p.Scalar(init=270)  # out_offset
        #        .set_bounds(lower=210, upper=330)
        #        .set_mutation(sigma=10),
    )

    budget = 1024  # Total number of evaluations to perform
    num_workers = 16  # Number of parallel workers to use

    # The remotely trainable net
    remoteNet = RemoteNet(max_workers=num_workers, local=False, nElo_target=-15)

    # Use TBPSA optimizer
    optimizer = ng.optimizers.TBPSA(
        instrumentation, budget=budget, num_workers=num_workers
    )

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        recommendation = optimizer.minimize(
            remoteNet.train_and_test_net, executor=executor
        )

    print("Final Recommended solution:", recommendation.value)
