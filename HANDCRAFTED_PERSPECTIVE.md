# Handcrafted perspective experiment

Branches start directly from freshly fetched upstream defaults. Engine and
trainer are pinned in both recipes:

* Stockfish: `2dca964965dbfc5b654f57377853b4282413c9c9`
* Trainer: `09c3a1a96f1b63869cc33567d3c7a59eaccebd63`

The full `threats.yaml` recipe starts fresh and preserves the established three
stages: epochs 0–100, 100–560 and 560–600; batch 131072 for the first two stages,
65536 only for the last. Later stages resume this experiment's checkpoint.
It retains the existing data, loss, optimizer schedule and long-run step counts.
The middle stage retains its three repetitions. Expanded full-run cache IDs are
`992b82b6bb4b`, `0289138cd966` and `49bbb3074688`.
The shared FC0 affine uses upstream recipe factorized weight decay 4e-5.
There are no MoE losses, codebooks, generators, reseeding or custom weight scales.
`testing.yaml` is the existing short smoke recipe with matching pins and a fresh
first stage. Neither recipe has been launched.

The network has 85 feasible handcrafted own-material buckets plus one derived
fallback per role. Widths are 1024/32/32, with independent 16-neuron role branches,
full concatenated inputs, bias-free FC0 skips and averaged STM-score readouts.
The FT uses strict opposing-heavy K32H3, independently of own-heavy routing.
Fallback parameters equally average exposed experts; a per-role training-exposure
mask redirects zero-exposure experts only at inference/export. Rare feasible
experts are retained. This is incompatible with upstream and MoE7 checkpoints;
never resume another experiment's model or checkpoint.

Build targets remain ordinary `build`, not PGO. Testing supplies the compatible
exported net explicitly; the upstream default embedded net is incompatible.
Upstream FEN admission rules are retained, including early material rejection.
The fallback does not broaden admission of arbitrary FENs.

The upstream reference engine/net remains an overall-system playing reference.
A match against it cannot isolate the contribution of routing: controlled
ablations must retain the exact FT semantics, data and training schedule, and
need separately trained compatible baselines. No playing strength is established.

The trainer [architecture and validation report](https://github.com/TonyCongqianWang/nnue-pytorch/blob/09c3a1a96f1b63869cc33567d3c7a59eaccebd63/docs/handcrafted-perspective.md)
contains the format, CPU tests, storage, matched kernel proxy timings and traffic
histograms. No local training/optimizer steps or remote jobs were run for this
implementation. CUDA training is unvalidated.
