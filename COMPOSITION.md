# Composition-conditioned material recipe

This isolated experiment starts from fresh default branches, without earlier
global-threat or pawn-frontier experiments. `threats.yaml` retains all baseline
PSQ, full-threat and pawn-pair features, L1=1024, L2=32, the baseline downstream
layers, binpacks, schedules, and training/finetuning stages.

Pinned source commits, published in dependency order:

- Engine: `TonyCongqianWang/Stockfish@afeed3ab470035d96ae3b82070c6349e0a9a7fe6`,
  based on `ff7cee4244ec96ea488c402669e3674c0574dce6`.
- Trainer: `TonyCongqianWang/nnue-pytorch@edd5b08d3bed867104b27abaeeb0268ac42a166b`,
  based on `b26145006e54e6461cb4b51944ab05cc4850c580`. Its separate material
  compatibility workflow pins the exact engine commit. Original training CI
  remains unchanged.
- Recipe base: `4d3d66b9f4fceba32f6b0c6692bb0353b1d0555e`.
- Reference engine: freshly fetched default `ff7cee4244ec96ea488c402669e3674c0574dce6`.

The feature string propagates through training, checkpoint conversion, neuron
optimization and cross-checking. Material has ten own/enemy presence bits and
1024 buckets. Actual counts multiply packed class rows; both perspectives share
the table. Per class, one shared base plus 511 corrections gives 512 independent
rows. Export folds them to int8 without additional base rows. The recipe sets
`composition-weight-decay: 4e-5` separately from spatial FT decay, shrinking rare
corrections toward shared material. This regularization is experimental and
has not been strength-tuned. The existing factorized dense decay stays 4e-5.

Actual full-width overhead: **5,242,880 parameters**, **20 MiB float32 parameter
storage**, **5 MiB exported bytes**, plus gradients/optimizer memory. Complete
synthetic exports with equal descriptions measured 136,247,590 baseline versus
141,490,470 material bytes. Accumulator-state and Finny-entry allocations did not
grow on the tested x86-64 build. Quiet incremental moves do not update material
vectors; cold/stale king cache entries may reconcile material during refresh.
Warm-cache synthetic update timings put quiet and king cases within about 2%
of baseline, and two capture cases at about +13% and +40%. This does not measure
full search throughput or establish any playing-strength improvement.

Validated independently of earlier branches:

- All 1024 buckets and perspective/mirroring/count parity across Python, the
  native loader, and Stockfish; full-width shared-base gradient checks.
- 5379 accumulator comparisons each in scalar and AVX2 builds, with lazy updates,
  undo, null moves, captures with and without presence changes, promotions,
  en passant, castling, king moves and refreshes; explicit quiet-update counters.
- Synthetic 1024-lane engine/trainer cross-check on 1024 positions with quantized
  differences below 1 cp; trainer → engine export is byte-identical.
- Complete short training/resume/quantization/serialization/neuron-permutation
  smoke at 128 lanes, then trained-network cross-check on 1024 positions against
  a matching scalar engine with quantized differences below 1 cp.
- Relevant trainer tests, Ruff, formatting, workflow/shell checks, recipe
  expansion/pipeline generation and nettest unit tests.

The environment has an 8 GiB memory limit. The full-width short training pipeline
was killed by that limit; it is not claimed to pass locally. A full trainer
pytest run reported 77 passes, 409 skips and one two-rank training SIGKILL; that
same test fails by SIGKILL on the untouched fresh trainer default. New compatibility
CI uses a 128-lane smoke while retaining the original baseline CI/default smoke
at 1024 lanes. Native and synthetic-network compatibility checks use 1024 lanes.

No full training, games, Stockfish bench, or strength tests were run. This file
and the recipe describe a reproducible experiment, not a trained or stronger net.
For commands and detailed cache/layout behavior, see the pinned repositories'
`tests/COMPOSITION.md`, `docs/composition.md`, and `scripts/test_composition.sh`.
