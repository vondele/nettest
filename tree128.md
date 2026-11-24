# Frozen tree128 recipes

All three recipes derive from upstream/main `threats.yaml` at
`6b75d7968ee9eb0a3cf98fd30db90cce37aedc28`. Training datasets, optimizer,
filtering, QAT warmup, loss, learning-rate/lambda schedules, lengths, seeds,
repetitions, conversion, optimization and crosscheck settings are preserved.
The reference is the current official Stockfish upstream/master pin
`3e8fde75412e7c47fc140cd82022a395bc681d92`, with its official network.

- `tree128.yaml`: 128 canonical leaf stacks, identity remap, ancestry factors
  at depths 2/4/6 (4/16/59 groups), plus master's global FC0 factorization.
- `tree128-unfactorized.yaml`: same tree, no extra group factors.
- `tree128-control.yaml`: master's 192 Cartesian stacks, identity remap, no
  extra group factors. Compare this with the unfactorized tree to isolate
  routing/count; compare the two tree arms to measure ancestry factorization.

Trainer/engine SHAs pin the local implementation commits. The optional `repo`
field points to an independent local git checkout, enabling normal recipe
fetch/build without publishing these commits. For a future remote run, publish
only when authorized, then replace `repo` with the published repository URL
and confirm the commit pins. `make_options` is included in the engine build
cache identity so the master control cannot reuse a tree executable.
Testing engines build without embedding and use the normal exported-network
EvalFile path. The unchanged official reference builds normally.

The normal conversion/FT optimization/crosscheck flow uses the same model
flags at every stage. Group factors coalesce at export; remap sampling is off
in every arm. The recipes have not been launched remotely.

`layer_stack_traffic.jsonl` records actual consumed leaf counts and separately
provider-delivered post-filter source counters (including prefetch). The
analysis objective's 95/5 weighting does not configure a training mixture.
Upstream tree acceptance and datasets are preserved; measure the resulting
source fraction before claiming it is 5%.

For inference performance use the engine's `tests/profile_tree128.py` and
`tests/bench_tree128.py` with a matching 128-stack net and a matching master
control. Recursive profile targets replace EXTRACXXFLAGS; pass layout/control
macros through CPPFLAGS. Do not profile with an incompatible default network.
