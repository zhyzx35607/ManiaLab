# V5 Full-Chart Remap Validation

Date: 2026-11-12

## Scope

This compares v4 (`sequence-window-v1`) and v5
(`sequence-window-ln-interactions-v1`) on fixed 10K note events and LN
endpoints. It evaluates lane assignment only; it does not measure note
transcription, LN duration, song-section planning, or play quality.

The alternate-split validation reports contain 14 independent 10K song groups.
To keep this first full-chart pass bounded, one map per group was selected
deterministically with seed `42`. Both models used the same 14 map IDs and
strength `0.15`. All 28 map runs passed timing, object, LN endpoint,
playability, and `.osu` round-trip checks. No test split was opened.

## Runtime

The v5 runtime previously encoded each candidate window independently,
recomputing the v1 prefix and LN hand-interaction state. It now reuses the
existing batched v1 prefix encoder and computes the 68 LN interaction values
for a batch of candidate windows.

- On 3,000 synthetic windows, batch encoding took `0.081s` versus `1.472s`
  for the training-time single-window encoder (`18.2x`); maximum feature
  difference was zero.
- Random legal state coverage checked 120 windows for every key count from
  6K through 10K; batch and single-window outputs matched exactly.
- On validation map `4993903`, v5 remap time fell from `236.710s` to
  `49.167s` (`4.8x`). All non-timing output fields matched exactly.
- In the 14-map comparison, mean prior time was `13.25s` for v4 and `18.89s`
  for v5; mean baseline time was `7.52s` and `7.63s`, respectively.

## Paired Results

Deltas are v5 minus v4. Values are song-group macro means; intervals use 5,000
paired song-group bootstrap samples with seed `20261112`. Metrics marked LN
only have 13 groups with LN events. These metrics describe source-map
similarity or pattern counts; they are not quality labels.

| Metric | v4 | v5 | Delta (95% group interval) |
|---|---:|---:|---:|
| Solver cost per note | 1.337523 | 1.337893 | +0.000370 [-0.001182, +0.002256] |
| Short jacks per 100 | 0.493492 | 0.502974 | +0.009482 [0, +0.021589] |
| Release repeats per 100 | 0.121926 | 0.129547 | +0.007620 [0, +0.022861] |
| Lanes changed from source | 0.335790 | 0.326451 | -0.009339 [-0.018561, -0.000687] |
| LN lanes changed from source | 0.440108 | 0.432523 | -0.007586 [-0.048209, +0.034585] |
| LN lane entropy | 0.881883 | 0.871122 | -0.010761 [-0.019321, -0.003674] |
| LN inverse-profile error | 0.351210 | 0.444573 | +0.093364 [0, +0.205987] |
| LN reverse-shield profile error | 1.286405 | 1.182187 | -0.104218 [-0.354362, +0.065129] |
| LN split-release profile error | 1.245121 | 1.165200 | -0.079921 [-0.406729, +0.250269] |
| LN release-press profile error | 4.773686 | 4.269544 | -0.504143 [-2.068921, +0.950171] |
| Same-hand hybrid profile error | 0.894298 | 0.999886 | +0.105588 [-0.026417, +0.276253] |

The v5 model did not show a balanced LN-layout improvement. It moved slightly
fewer notes away from the source chart, but increased the short-jack and
inverse-profile point estimates. Several other LN profile estimates improved,
with intervals spanning zero. The entropy change means LN placements became
more concentrated across lanes; this experiment cannot determine whether that
is desirable. The model remains disconnected from default generation.

## Reproduction

```powershell
.\.venv-audio\Scripts\python.exe scripts\evaluate_sequence_prior_remap.py --report test-data\sequence-prior-ln-transition-alt-split-20261112\report.json --model test-data\sequence-prior-ln-transition-alt-split-20261112\model.json --output test-data\sequence-prior-remap-alt-v4-validation-pergroup-20261112.json --keys 10 --split validation --strength 0.15 --seed 42 --limit-per-group 1
.\.venv-audio\Scripts\python.exe scripts\evaluate_sequence_prior_remap.py --report test-data\sequence-prior-ln-interactions-alt-split-20261112\report.json --model test-data\sequence-prior-ln-interactions-alt-split-20261112\model.json --output test-data\sequence-prior-remap-alt-v5-validation-pergroup-20261112.json --keys 10 --split validation --strength 0.15 --seed 42 --limit-per-group 1
```

Full per-map JSON reports and models remain local ignored artifacts under
`test-data`; they are not included in source releases.
