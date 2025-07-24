from firecrest_executor import FirecrestExecutor
import nevergrad as ng


# the script that will be executed in the container on the remote node(s)
def train_and_test_net(pow_exp, random_fen_skipping):
    import re
    import subprocess

    pattern = re.compile(
        # r"^\s*Result\s*:\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
        r"^\s*Elo\s*:\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
    )

    opt = f"""
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
                     - --end-lambda=1.0
                     - --gamma=0.9942746303116422
                     - --lr=0.0012181558724738395
                     - --in-scaling=320.15446837357985
                     - --out-scaling=396.8462790115712
                     - --in-offset=278.73702665289676
                     - --out-offset=234.10701378957856
                     - --pow-exp={pow_exp}
                     - --random-fen-skipping={random_fen_skipping}
                     - --simple-eval-skipping=920
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
                  max_rounds: 100000
           """

    print("============ opt.yaml ==========")
    with open("opt.yaml", "w") as f:
        f.write(opt)
    print(opt)
    print("============ opt.yaml ==========")

    # translate opt.yaml into execute.sh
    process = subprocess.Popen(
        [
            "python",
            "-u",
            "nettest/generate_pipeline.py",
            "opt.yaml",
            "./execute.yaml",
            "/workspace",
            "./f7t-06",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if process.stdout is not None:
        for line in process.stdout:
            print(line, end="", flush=True)
        process.stdout.close()
    process.wait()

    # Execute the generated script and capture the output
    print("============ Executing script ==========")
    result = None
    process = subprocess.Popen(
        ["bash", "execute.sh"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # match the output against the pattern to extract the Elo value
    if process.stdout is not None:
        for line in process.stdout:
            print(line, end="", flush=True)
            match = pattern.match(line)
            if match:
                # Minus the Elo, since we minimize
                result = -float(match.group(1))
        process.stdout.close()
    process.wait()

    if result is not None:
        return result
    else:
        raise ValueError("Failed to convert output, no matching pattern found")


instrumentation = ng.p.Instrumentation(
    ng.p.Scalar(init=2.5).set_bounds(lower=2.0, upper=3.0).set_mutation(sigma=0.15),
    ng.p.Scalar(init=10)
    .set_bounds(lower=2, upper=20)
    .set_mutation(sigma=2)
    .set_integer_casting(),
)
budget = 64  # Total number of evaluations to perform
num_workers = 8  # Number of parallel workers to use

# Use TBPSA optimizer
optimizer = ng.optimizers.TBPSA(instrumentation, budget=budget, num_workers=num_workers)

# Use the created executor to execute scripts
with FirecrestExecutor(
    working_dir="/users/vjoost/fish/workspace/",
    sbatch_options=[
        "--job-name=FirecrestExecutor",
        "--time=12:00:00",
        "--nodes=1",
        "--partition=normal",
    ],
    srun_options=["--environment=/users/vjoost/fish/workspace/nettest.toml"],
    sleep_interval=5,
) as executor:
    recommendation = optimizer.minimize(train_and_test_net, executor=executor)

print("Final Recommended solution:", recommendation.value)
