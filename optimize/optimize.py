from firecrest_executor import FirecrestExecutor
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from nettest import execute
import nevergrad as ng
import yaml


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
            )
        self.nElo_target = nElo_target

    # the recipe to optimize
    def train_and_test_net(
        self,
        pow_exp,
        end_lambda,
        in_scaling,
        out_scaling,
        in_offset,
        out_offset,
    ):
        print(
            "Starting:",
            pow_exp,
            end_lambda,
            in_scaling,
            out_scaling,
            in_offset,
            out_offset,
            flush=True,
        )
        recipe_str = f"""
training:
  steps:
    - convert:
        binpack: official-stockfish/master-binpacks/fishpack32.binpack
        checkpoint2nnue:
          - --features=HalfKAv2_hm^
          - --ft_compression=leb128
        optimize:
          - --features=HalfKAv2_hm
          - --ft_optimize_count=100000
          - --ft_optimize
          - --ft_compression=leb128
      datasets:
        - hf:
            owner: official-stockfish
            repo: master-binpacks
        - hf:
            owner: vondele
            repo: from_classical
      run:
        binpacks:
          - vondele/from_classical/from_classical_05_pv-2_diff-100_nodes-5000.binpack
          - vondele/from_classical/from_classical_04_pv-2_diff-100_nodes-5000.binpack
          - vondele/from_classical/from_classical_03_pv-2_diff-100_nodes-5000.binpack
        max_epochs: 800
        repetitions: 3
        other_options:
          - --batch-size=16384
          - --features=HalfKAv2_hm^
          - --start-lambda=1.0
          - --end-lambda={end_lambda}
          - --gamma=0.997
          - --lr=0.0012
          - --in-scaling={in_scaling}
          - --out-scaling={out_scaling}
          - --in-offset={in_offset}
          - --out-offset={out_offset}
          - --pow-exp={pow_exp}
          - --random-fen-skipping=10
        resume: none
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
      sha: 64e8e1b247367018ebbce15f0d7cf5a56c7eaf56
  testing:
    code:
      owner: official-stockfish
      sha: 64e8e1b247367018ebbce15f0d7cf5a56c7eaf56
  fastchess:
    code:
      owner: Disservin
      sha: 7676d4961518a7a0156bcd200149acc8c921f977
    options:
       tc: 10+0.1
       hash: 16
    sprt:
       nElo_interval_midpoint: {self.nElo_target}
       nElo_interval_width: 4
       max_rounds: 160000
               """

        recipe = yaml.safe_load(recipe_str)

        nElo = execute(
            recipe=recipe,
            executor=self.executor,
        )

        if nElo is None:
            # TODO ... better error handling possible ?
            print("Something went wrong during evaluation .... continuing with lower bound estimate"
            nElo = self.nElo_target - 10

        print(
            "Done:",
            pow_exp,
            end_lambda,
            in_scaling,
            out_scaling,
            in_offset,
            out_offset,
            nElo,
            flush=True,
        )

        if nElo > self.nElo_target:
            self.nElo_target = nElo

        return -nElo

if __name__ == "__main__":
    instrumentation = ng.p.Instrumentation(
        ng.p.Scalar(init=2.173) # pow_exp
        .set_bounds(lower=2.0, upper=3.0)
        .set_mutation(sigma=0.15),
        ng.p.Scalar(init=0.7) # end_lambda
        .set_bounds(lower=0.5, upper=1.0)
        .set_mutation(sigma=0.05),
        ng.p.Scalar(init=281) # in_scaling
        .set_bounds(lower=220, upper=360)
        .set_mutation(sigma=20),
        ng.p.Scalar(init=440) # out_scaling
        .set_bounds(lower=360, upper=500)
        .set_mutation(sigma=20),
        ng.p.Scalar(init=291) # in_offset
        .set_bounds(lower=240, upper=340)
        .set_mutation(sigma=20),
        ng.p.Scalar(init=218) # out_offset
        .set_bounds(lower=180, upper=275)
        .set_mutation(sigma=20),
    )

    budget = 128  # Total number of evaluations to perform
    num_workers = 16  # Number of parallel workers to use

    # The remotely trainable net
    remoteNet = RemoteNet(max_workers=num_workers, local=False, nElo_target=-170)

    # Use TBPSA optimizer
    optimizer = ng.optimizers.TBPSA(
        instrumentation, budget=budget, num_workers=num_workers
    )

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        recommendation = optimizer.minimize(
            remoteNet.train_and_test_net, executor=executor
        )

    print("Final Recommended solution:", recommendation.value)
