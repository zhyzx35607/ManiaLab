# LN Transition Hard-Negative Experiment

Date: 2026-11-12

## Scope

The experiment trains the four-row lane-ranking prior with feasible alternatives
that change a local LN interaction. Note onsets and LN endpoints stay fixed.
The model does not choose note events, infer instrument roles, plan sections, or
decide LN duration. It remains disconnected from default generation.

The run used the 7,907-map 6K-10K corpus and the isolated
`sequence-window-cache-v4-ln-transition` cache. It trained all key counts in
one network, rotating optimization steps evenly across available key counts
and balancing LN/tap examples within each key count. The checkpoint was chosen
at epoch 20 of 24 using validation loss. Export maximum error was
`3.81e-5`.

For 10K, the retained data contained 27,764 train windows from 83 groups,
3,405 validation windows from 13 groups, and 3,694 test windows from 12 groups.
On the 10K validation set, the selected model scored 75.86% over 6,810 pairs,
76.01% over LN pairs, and 75.76% over tap pairs. Those are hard-negative
ranking scores, not play-quality ratings.

## Paired Comparison

Each comparison scores the v3 model and v4 model on the exact same sampled
windows and alternatives. Deltas are v4 minus v3, macro-averaged by independent
song group; intervals use 5,000 paired group-bootstrap samples with seed
`20261112`.

The 10K test results were:

| Candidate set | Groups | Overall delta (95% CI) | LN delta (95% CI) | Tap delta (95% CI) |
|---|---:|---:|---:|---:|
| v3 random feasible | 12 | -0.31 pp (-1.41 to +0.75) | +0.99 pp (-0.67 to +2.56) | -0.02 pp (-2.35 to +2.43) |
| v4 LN-transition hard | 12 | -0.68 pp (-3.33 to +1.43) | +2.01 pp (-1.82 to +5.30) | -1.98 pp (-4.86 to +0.35) |

The validation comparison did not show a consistent gain either:

| Candidate set | Groups | Overall delta (95% CI) | LN delta (95% CI) | Tap delta (95% CI) |
|---|---:|---:|---:|---:|
| v3 random feasible | 13 | -0.87 pp (-2.14 to +0.35) | -1.03 pp (-4.45 to +1.80) | -1.38 pp (-2.96 to +0.09) |
| v4 LN-transition hard | 13 | -1.37 pp (-2.94 to +0.33) | -0.67 pp (-2.67 to +1.30) | -1.78 pp (-4.20 to +0.54) |

All 10K intervals include zero. The v4 model therefore has no demonstrated
balanced ranking improvement over v3, and its point estimates trade a small LN
gain on the hard test candidates for lower tap and overall scores. The 6K-9K
results do not change that conclusion; 8K and 9K have few independent groups.
Do not connect this model to generation or increase its runtime weight.

## 10K Specialist

A 10K-only model was trained on the same v4 hard-negative cache. Validation
selected epoch 4 of 24. On identical 10K validation candidates, compared with
the joint v4 model, the specialist's paired changes were:

| Metric | Delta (95% group-bootstrap CI) |
|---|---:|
| Overall | -2.30 pp (-3.58 to -1.02) |
| LN | +0.57 pp (-1.08 to +2.47) |
| Tap | -4.08 pp (-6.06 to -2.42) |

The LN interval includes zero while overall and tap ranking decrease. This
does not support connecting the specialist to generation. The comparison uses
2,668 LN pairs from 12 groups and 4,142 tap pairs from 12 groups; the unit of
the interval is the independent song group. The report is retained locally at
`test-data/sequence-prior-cross-v1-validation-v4-10k-specialist.json`.

## Next Experiment

The next validation hypothesis adds explicit LN interaction summaries while
preserving the original 382 input values as an exact prefix. The new
`sequence-window-ln-interactions-v1` schema adds 17 per-row values for active
LN combinations, hybrid taps/heads, release-press interactions, inverse,
reverse-shield, shield, and near-release behavior. It reads only the supplied
window and per-lane left-edge state, so training and generation can encode the
same information; older LN-release history not present in that runtime state
is excluded from both the v5 feature input and its hard-negative transition
signature, keeping training labels within the runtime-visible context. A
separate v5 cache prevents mixing these features with existing data.

The schema has passed focused training/runtime/evaluator tests and a small
train/export/cross-schema validation smoke. That smoke is only a pipeline
check: it used at most two maps per key and split and cannot establish a
ranking or play-quality gain. Full v5 training and paired 10K validation remain
the next steps. Do not select from test or connect this model to generation
unless validation demonstrates a balanced improvement. LN duration and
whole-chart playability remain separate unaddressed tasks.

The reports, models, caches, and comparison JSON files are ignored local
artifacts under `test-data`; none are included in the repository.
