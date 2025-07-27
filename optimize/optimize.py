from firecrest_executor import FirecrestExecutor
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from nettest import execute
import nevergrad as ng
import yaml


class RemoteNet:
    def __init__(self, max_workers=16, local=False):
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

    # the recipe to optimize
    def train_and_test_net(
        self,
        pow_exp,
        end_lambda,
        in_scaling,
        out_scaling,
        in_offset,
        out_offset,
        simple_eval_skipping,
    ):
        recipe_str = f"""
               #
               # nnue-pytorch training steps
               #
               training:
                 steps:
               #
               # attempt to mimic SF cb4a62311985f685ba6f5457851527a3289073e6 data sets missing or approximate..
               #
                   - datasets:
                       - hf:
                          owner: official-stockfish
                          repo: master-binpacks
                       - hf:
                          owner: linrock
                          repo: dual-nnue
                       - hf:
                          owner: linrock
                          repo: test80-2022
                       - hf:
                          owner: linrock
                          repo: test80-2023
                       - hf:
                          owner: linrock
                          repo: test80-2024
                     trainer:
                       owner: vondele
                       sha: 1600ce0c04380ae15c3b0912c3609fb37f815431
                     run:
                       resume: none
                       max_epochs: 400
                       binpacks:
                         - linrock/dual-nnue/hse-v1/dfrc99-16tb7p-eval-filt-v2.min.high-simple-eval-1k.min-v2.binpack # TODO only an approx /data/dfrc99-16tb7p.v2.min.binpack?
                         - linrock/dual-nnue/hse-v1/leela96-filt-v2.min.high-simple-eval-1k.min-v2.binpack
                         - linrock/dual-nnue/hse-v1/test60-novdec2021-12tb7p-filter-v6-dd.min-mar2023.unmin.high-simple-eval-1k.min-v2.binpack
                         - linrock/dual-nnue/hse-v1/test77-nov2021-2tb7p.no-db.min.high-simple-eval-1k.min-v2.binpack
                         - linrock/dual-nnue/hse-v1/test77-dec2021-16tb7p.no-db.min.high-simple-eval-1k.min-v2.binpack
                         - linrock/dual-nnue/hse-v1/test77-jan2022-2tb7p.high-simple-eval-1k.min-v2.binpack
                         - linrock/dual-nnue/hse-v1/test78-jantomay2022-16tb7p-filter-v6-dd.min-mar2023.unmin.high-simple-eval-1k.min-v2.binpack
                         - linrock/dual-nnue/hse-v1/test78-juntosep2022-16tb7p-filter-v6-dd.min-mar2023.unmin.high-simple-eval-1k.min-v2.binpack
                         - linrock/dual-nnue/hse-v1/test79-apr2022-16tb7p.min.high-simple-eval-1k.min-v2.binpack
                         - linrock/dual-nnue/hse-v1/test79-may2022-16tb7p-filter-v6-dd.min-mar2023.unmin.high-simple-eval-1k.min-v2.binpack
                         - linrock/dual-nnue/hse-v1/test80-apr2022-16tb7p.min.high-simple-eval-1k.min-v2.binpack
                         - linrock/dual-nnue/hse-v1/test80-may2022-16tb7p.high-simple-eval-1k.min-v2.binpack
                         - linrock/dual-nnue/hse-v1/test80-jun2022-16tb7p-filter-v6-dd.min-mar2023.unmin.high-simple-eval-1k.min-v2.binpack
                         - linrock/dual-nnue/hse-v1/test80-jul2022-16tb7p.v6-dd.min.high-simple-eval-1k.min-v2.binpack
                         - linrock/dual-nnue/hse-v1/test80-sep2022-16tb7p-filter-v6-dd.min-mar2023.unmin.high-simple-eval-1k.min-v2.binpack
                         - linrock/dual-nnue/hse-v1/test80-nov2022-16tb7p-v6-dd.min.high-simple-eval-1k.min-v2.binpack
               #
                         - linrock/test80-2022/test80-2022-08-aug-16tb7p.v6-dd.min.binpack
                         - linrock/test80-2022/test80-2022-10-oct-16tb7p.v6-dd.binpack
                         - linrock/test80-2022/test80-2022-12-dec-16tb7p.min.binpack
               #
                         - linrock/test80-2023/test80-2023-01-jan-16tb7p.v6-sk20.min.binpack
                         - linrock/test80-2023/test80-2023-02-feb-16tb7p.v6-dd.min.binpack # test80-2023-02-feb-16tb7p.v6-sk20.min.binpack
                         - linrock/test80-2023/test80-2023-03-mar-2tb7p.v6-sk16.min.binpack
                         - linrock/test80-2023/test80-2023-04-apr-2tb7p.v6-sk16.min.binpack
                         - linrock/test80-2023/test80-2023-05-may-2tb7p.v6.min.binpack
                         - linrock/test80-2023/test80-2023-06-jun-2tb7p.min-v2.binpack
                         - linrock/test80-2023/test80-2023-07-jul-2tb7p.min-v2.binpack
                         - linrock/test80-2023/test80-2023-08-aug-2tb7p.v6.min.binpack
                         - linrock/test80-2023/test80-2023-09-sep-2tb7p.binpack # hse-v6.binpack ?
                         - linrock/test80-2023/test80-2023-10-oct-2tb7p.binpack # idem
                         - linrock/test80-2023/test80-2023-11-nov-2tb7p.min-v2.binpack # idem
                         - linrock/test80-2023/test80-2023-12-dec-2tb7p.min-v2.binpack
               #
                         - linrock/test80-2024/test80-2024-01-jan-2tb7p.min-v2.v6.binpack
                         - linrock/test80-2024/test80-2024-02-feb-2tb7p.min-v2.v6.binpack
                         - linrock/test80-2024/test80-2024-03-mar-2tb7p.min-v2.v6.binpack
                       other_options:
                         - --batch-size=16384
                         - --features=HalfKAv2_hm^
                         - --l1=128
                         - --no-wld-fen-skipping
                         - --start-lambda=1.0
                         - --end-lambda={end_lambda}
                         - --gamma=0.9942746303116422
                         - --lr=0.0012181558724738395
                         - --in-scaling={in_scaling}
                         - --out-scaling={out_scaling}
                         - --in-offset={in_offset}
                         - --out-offset={out_offset}
                         - --pow-exp={pow_exp}
                         - --random-fen-skipping=3
                         - --simple-eval-skipping={simple_eval_skipping}
                         - --compile-backend=cudagraphs
                     convert:
                       binpack: official-stockfish/master-binpacks/fishpack32.binpack
                       optimize:
                          - --features=HalfKAv2_hm
                          - --l1=128
                          - --ft_optimize_count=100000
                          - --ft_optimize
                          - --ft_compression=leb128
                       checkpoint2nnue:
                          - --features=HalfKAv2_hm^
                          - --l1=128
                          - --ft_compression=leb128
               
               #
               # Final testing stage using fastchess
               #
               testing:
                 reference:
                   code:
                     owner: vondele
                     sha: 40a8ec1dd5ca4c53406eae8861bd51ba907bc69a
                 testing:
                   code:
                     owner: vondele
                     sha: 40a8ec1dd5ca4c53406eae8861bd51ba907bc69a
                   options:
                     - smallNetThreshold=932
                 fastchess:
                   code:
                     owner: Disservin
                     sha: 6072a9bd24c07b300e3feb70ce7f77c2be8537cc
                   options:
                      tc: 10+0.1
                      hash: 16
                      evalfile: small
                   sprt:
                      # midpoint will be the target to reach, relative to reference, to have a passing CI
                      nElo_interval_midpoint: 1.0
                      nElo_interval_width: 2.0
                      max_rounds: 200000
               """

        recipe = yaml.safe_load(recipe_str)

        Elo = execute(
            recipe=recipe,
            executor=self.executor,
        )

        if Elo is None:
            raise ValueError(
                "execute returned None, something went wrong during execution."
            )

        return -Elo


if __name__ == "__main__":
    instrumentation = ng.p.Instrumentation(
        ng.p.Scalar(init=2.5).set_bounds(lower=2.0, upper=3.0).set_mutation(sigma=0.15),
        ng.p.Scalar(init=0.8).set_bounds(lower=0.5, upper=1.0).set_mutation(sigma=0.05),
        ng.p.Scalar(init=320).set_bounds(lower=280, upper=360).set_mutation(sigma=20),
        ng.p.Scalar(init=380).set_bounds(lower=360, upper=440).set_mutation(sigma=20),
        ng.p.Scalar(init=280).set_bounds(lower=240, upper=320).set_mutation(sigma=20),
        ng.p.Scalar(init=235).set_bounds(lower=195, upper=275).set_mutation(sigma=20),
        ng.p.Scalar(init=920).set_bounds(lower=800, upper=1050).set_mutation(sigma=50).set_integer_casting(),
    )

    budget = 256  # Total number of evaluations to perform
    num_workers = 128  # Number of parallel workers to use

    # The remotely trainable net
    remoteNet = RemoteNet(max_workers=num_workers, local=False)

    # Use TBPSA optimizer
    optimizer = ng.optimizers.TBPSA(
        instrumentation, budget=budget, num_workers=num_workers
    )

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        recommendation = optimizer.minimize(
            remoteNet.train_and_test_net, executor=executor
        )

    print("Final Recommended solution:", recommendation.value)
