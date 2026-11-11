# 10K LN Warm-Start Replication

Date: 2026-11-12

## Question

The joint v5 LN-interaction model showed a 10K LN ranking gain on its
validation set, but that gain did not replicate on its held-out test. A
10K-only model trained from random initialization also performed worse than
the joint v5 model. This follow-up tests whether a short, low-learning-rate
10K fine-tune starting from a joint model is more stable.

All comparisons still use fixed authored events, fixed LN endpoints, and
feasible lane alternatives. They do not evaluate note selection, LN duration,
instrument roles, whole-chart layout, or playability.

## Initial Validation

On the original split, the warm-start model was compared with the v4 baseline
using the same 45 10K validation maps, 13 groups, and 3,405 windows. The
warm-start model improved LN ranking by `+2.51 pp` (95% group-bootstrap CI
`[+0.42, +4.49]`); overall changed by `+1.64 pp` (`[-0.23, +3.36]`) and tap
by `+0.90 pp` (`[-1.51, +3.63]`). This was a validation-only signal, not a
held-out confirmation.

## Fresh-Split Replication

Training and evaluation were repeated with deterministic alternate song-group
split seed `20261112`. The source corpus has 7,907 maps; the new 10K validation
set contains 150 maps from 14 groups and 6,909 windows, including 5,226 LN
and 8,592 tap candidate pairs. All models were trained from scratch within
this split except the warm-start model, which starts from the joint v5 model
trained on this same split. Final training reports omitted test metrics.

The 10K joint v5 and warm-start validation results were compared with the
freshly trained v4 baseline on identical v5 candidates. Deltas are the second
model minus the first, macro-averaged by independent song group; intervals
use 5,000 paired group-bootstrap samples with seed `20261112`.

| Comparison | Overall delta (95% CI) | LN delta (95% CI) | Tap delta (95% CI) |
|---|---:|---:|---:|
| Joint v5 vs v4 | +0.56 pp (-0.88 to +2.47) | +1.35 pp (-0.58 to +3.37) | -0.52 pp (-1.73 to +0.76) |
| Warm-start vs v4 | +0.80 pp (-0.61 to +2.76) | +1.17 pp (-0.76 to +3.20) | -0.17 pp (-1.50 to +1.17) |
| Warm-start vs joint v5 | +0.24 pp (-0.33 to +0.86) | -0.18 pp (-0.80 to +0.36) | +0.35 pp (-0.45 to +1.17) |

Each comparison covers 14 total groups; 13 groups contribute LN and tap
candidates. The LN intervals for both v5 comparisons with v4 include zero,
and warm-start is not distinguishable from the joint model. The earlier
positive validation estimate therefore did not reproduce on this new
validation split.

The alternate split contains seven 10K test groups, six with retained LN
windows. Their metadata overlaps the original evaluated 10K test set by one
group. No test scores were calculated for this replication because validation
showed no clear gain. Re-splitting the same corpus does not create a completely
new, untouched test set; these groups must not be presented as such.

## Decision

The 10K-only random-initialized model is rejected: on the original validation
set it was worse than joint v5 by `-4.69 pp` overall (`[-6.09, -3.28]`),
`-2.33 pp` for LN (`[-4.09, -0.77]`), and `-6.10 pp` for taps
(`[-8.31, -4.15]`).

Warm-start was less damaging than training a 10K specialist from scratch, but
the fresh-split comparison does not show a reliable gain over either v4 or
joint v5. Neither warm-start nor the LN-interaction model is connected to
default generation. A future change should first improve the training target
or add independent 10K groups, then repeat a pre-registered group split and
evaluate its test set once.

## Reproduction

The trainer supports an optional deterministic alternate split, compatible
warm-start model, learning rate, and test-metric omission. Defaults preserve
the prior training behavior. Skipping test evaluation omits test scoring,
but preparation still loads test caches and records split counts. The v4
and v5 caches are unchanged because their contents are per-map features and
alternatives; split membership is recorded in each report.

Warm-start requires the source model's companion report (`report.json` next
to the model, or `--initial-report`). The complete source manifest is rehashed
and must exactly match the current manifest, including song-group splits.
Schema, feature names, and training keys must agree between the source model
and report; joint-to-10K fine-tuning is supported. Model and report hashes are
recorded. Further warm-starts require a matching ancestor-manifest proof in
both artifacts. Legacy warm-start outputs without that proof cannot be used
as new training sources. These checks reject cross-split reuse of weights.

```powershell
.\.venv-audio\Scripts\python.exe scripts\train_sequence_prior.py --threads 2 --split-seed 20261112 --skip-test-evaluation --negative-policy ln-transition-hard --feature-schema sequence-window-v1 --cache test-data\corpus-expanded\sequence-window-cache-v4-ln-transition --output test-data\sequence-prior-ln-transition-alt-split-20261112

.\.venv-audio\Scripts\python.exe scripts\train_sequence_prior.py --threads 2 --split-seed 20261112 --skip-test-evaluation --negative-policy ln-transition-hard --feature-schema sequence-window-ln-interactions-v1 --cache test-data\corpus-expanded\sequence-window-cache-v5-ln-interactions --output test-data\sequence-prior-ln-interactions-alt-split-20261112

.\.venv-audio\Scripts\python.exe scripts\train_sequence_prior.py --threads 2 --keys 10 --epochs 8 --steps 40 --learning-rate 0.0001 --split-seed 20261112 --negative-policy ln-transition-hard --feature-schema sequence-window-ln-interactions-v1 --initial-model test-data\sequence-prior-ln-interactions-alt-split-20261112\model.json --skip-test-evaluation --cache test-data\corpus-expanded\sequence-window-cache-v5-ln-interactions --output test-data\sequence-prior-ln-interactions-10k-warmstart-alt-split-20261112
```

The model reports, caches, and paired comparison JSON files stay in ignored
local `test-data`; no generated weights or song data enter the source tree.
