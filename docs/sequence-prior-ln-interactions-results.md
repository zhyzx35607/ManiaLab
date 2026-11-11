# LN Interaction Feature Experiment

Date: 2026-11-12

## Scope

This experiment adds explicit lane-by-lane LN interaction summaries to the
four-row sequence-prior input. The original 382 features remain an exact prefix;
the v5 schema adds 68 values (17 per row) for hold occupancy, hand combinations,
hybrid patterns, release-press interactions, inverse, reverse-shield, shield,
and near-release behavior.

The model ranks feasible lane assignments for fixed authored note events and
fixed LN endpoints. It does not choose note events, assign instrument roles,
plan song sections, or determine LN length. It remains disconnected from
default generation. Results therefore describe local lane ranking, not chart
quality or playability.

## Training

The joint 6K-10K model used all 7,907 maps from the expanded corpus, the
versioned v5 cache, and the `ln-transition-hard` negative policy. The model was
selected at epoch 18 of 24 by validation loss. The exported network's maximum
absolute error was `3.81e-5`.

For 10K, the prepared data contained 27,764 train windows from 83 song groups,
3,405 validation windows from 13 groups, and 3,694 test windows from 12 groups.
Validation retained 1,360 of 1,461 sampled LN windows (93.1%); 101 sampled
windows could not produce the required hard negatives. After the song-group
window budget, paired validation evaluated 1,334 LN windows and 2,071 tap
windows. The held-out 10K test contained 1,561 LN windows and 2,133 tap
windows.

The full report and model are local ignored artifacts at
`test-data/sequence-prior-ln-interactions-v1`. The cache is under
`test-data/corpus-expanded/sequence-window-cache-v5-ln-interactions`.

## Paired Validation

The v4 and v5 models were scored on identical v5 validation candidates. The
v4 model reads the exact 382-feature prefix. Deltas are v5 minus v4 and are
macro-averaged over independent song groups; confidence intervals use 5,000
paired group-bootstrap samples with seed `20261112`.

| Keys | Overall delta (groups, 95% CI) | LN delta (groups, 95% CI) | Tap delta (groups, 95% CI) |
|---|---:|---:|---:|
| 6K | -0.54 pp (31, -1.11 to +0.05) | -0.49 pp (28, -1.56 to +0.63) | -0.48 pp (31, -1.30 to +0.34) |
| 7K | -0.07 pp (155, -0.41 to +0.27) | +0.60 pp (141, -0.02 to +1.22) | -0.47 pp (153, -1.14 to +0.16) |
| 8K | -0.04 pp (8, -0.90 to +0.83) | -1.38 pp (5, -7.87 to +4.04) | -0.14 pp (8, -1.27 to +0.92) |
| 9K | -0.43 pp (6, -1.33 to +0.57) | -0.82 pp (5, -2.05 to +0.41) | +0.04 pp (6, -1.63 to +2.09) |
| 10K | +1.14 pp (13, -0.89 to +3.06) | +2.61 pp (12, +0.23 to +5.20) | +0.04 pp (12, -2.47 to +3.02) |

The 10K LN validation interval excludes zero, with no negative point estimate
for overall or tap ranking. This justified one paired held-out 10K test; the
other key counts did not show a comparable validation signal. The number of
independent groups, not the number of candidate pairs, is the unit used for
the intervals.

## Held-Out 10K Test

The one-time paired comparison covered 48 maps, 12 independent song groups,
and 3,694 windows. Ten groups contributed LN candidates.

| Metric | Delta (95% group-bootstrap CI) |
|---|---:|
| Overall | +0.26 pp (-1.67 to +2.68) |
| LN | -0.07 pp (-2.61 to +3.44) |
| Tap | +0.87 pp (-1.31 to +3.21) |

All intervals include zero. The validation LN gain did not replicate on the
held-out test. These results do not demonstrate a generalization improvement,
so the v5 model is not connected to generation. The smaller 8K/9K validation
sets are especially uncertain, and the study does not establish a 6K-10K
improvement.

## Reproduction

Full training:

```powershell
.\.venv-audio\Scripts\python.exe scripts\train_sequence_prior.py --threads 2 --negative-policy ln-transition-hard --feature-schema sequence-window-ln-interactions-v1 --cache test-data\corpus-expanded\sequence-window-cache-v5-ln-interactions --output test-data\sequence-prior-ln-interactions-v1
```

Paired validation:

```powershell
.\.venv-audio\Scripts\python.exe scripts\evaluate_sequence_prior_cross_model.py --report-a test-data\sequence-prior-ln-transition-v1\report.json --model-a test-data\sequence-prior-ln-transition-v1\model.json --report-b test-data\sequence-prior-ln-interactions-v1\report.json --model-b test-data\sequence-prior-ln-interactions-v1\model.json --sample-report test-data\sequence-prior-ln-interactions-v1\report.json --cache test-data\corpus-expanded\sequence-window-cache-v5-ln-interactions --keys 6 7 8 9 10 --split validation --label-a v4-ln-transition --label-b v5-ln-interactions --output test-data\sequence-prior-cross-v5-validation-6-10.json
```

Paired held-out 10K test:

```powershell
.\.venv-audio\Scripts\python.exe scripts\evaluate_sequence_prior_cross_model.py --report-a test-data\sequence-prior-ln-transition-v1\report.json --model-a test-data\sequence-prior-ln-transition-v1\model.json --report-b test-data\sequence-prior-ln-interactions-v1\report.json --model-b test-data\sequence-prior-ln-interactions-v1\model.json --sample-report test-data\sequence-prior-ln-interactions-v1\report.json --cache test-data\corpus-expanded\sequence-window-cache-v5-ln-interactions --keys 10 --split test --label-a v4-ln-transition --label-b v5-ln-interactions --output test-data\sequence-prior-cross-v5-test-10k.json
```

Models, caches, and comparison JSON files are ignored local artifacts under
`test-data`; they are not included in the source repository.
