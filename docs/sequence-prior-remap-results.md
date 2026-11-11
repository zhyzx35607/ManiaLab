# Sequence-prior full-chart remap validation

Updated: 2026-11-11

## Protocol

This experiment evaluates only lane assignment for fixed 10K note events. It
keeps each source note, start time, end time, timing point, and LN duration
unchanged. It does not evaluate audio transcription or whether a chart feels
good to play.

The paired validation set contains 45 maps from 13 song groups; 12 groups have
LN notes. Each strength uses the same maps, seed, model, and complete candidate
space (`score_candidate_cap=1024`). Values below are sequence-prior minus
baseline, macro-averaged by song group. Intervals are 95% song-group bootstrap
intervals without a multiple-metric significance correction.

`cost_per_note` is the layout solver's ergonomic heuristic, not a quality
label. `lane_change_rate` measures how many assigned lanes differ from the
original map; a higher value means less reference similarity. LN profile error
metrics compare the generated pattern counts with the original map; they are
structural proxies, not musical judgments.

## Results

| Strength | Metric delta | 95% group interval | Reading |
|---:|---|---|---|
| 0.15 | `cost_per_note` -0.001384 | [-0.003085, 0.000224] | Small change; interval includes 0 |
| 0.15 | `lane_change_rate` +0.080562 | [0.070448, 0.090593] | More assignments differ from reference |
| 0.15 | `ln_reference_lane_change_rate` +0.021291 | [-0.127421, 0.113457] | No clear LN lane-change difference |
| 0.15 | `ln_inverse_profile_error` -0.032204 | [-0.156698, 0.072554] | No clear profile improvement |
| 0.15 | `ln_reverse_shield_profile_error` +0.112701 | [0.000000, 0.311252] | Point estimate is worse |
| 0.35 | `cost_per_note` -0.004630 | [-0.006457, -0.002829] | Lower heuristic cost |
| 0.35 | `lane_change_rate` +0.250791 | [0.234550, 0.265903] | Much more deviation from reference |
| 0.35 | `ln_reference_lane_change_rate` +0.261869 | [0.213838, 0.305008] | LN assignments move substantially |
| 0.35 | `ln_inverse_profile_error` +0.178843 | [0.000000, 0.501917] | LN inverse profile is worse |
| 0.35 | `ln_reverse_shield_profile_error` +0.175856 | [-0.000796, 0.472561] | No demonstrated improvement |

At strength 0.15, mean remap time was 8.65 seconds per chart versus 4.94
seconds for baseline (1.75x). At strength 0.35, it was 10.07 versus 4.93
seconds (2.04x).

## Decision

Neither completed strength shows a balanced LN placement improvement.
Strength 0.35 improves the solver's heuristic cost while moving many more notes
away from the source map and worsening the LN inverse profile. Neither value is
frozen for the independent test split. Strength 0.60 remains to be evaluated
on the same validation set.

Even a favorable remap result would not validate LN lengths, because this
experiment deliberately holds all LN endpoints fixed. Human review and actual
play testing remain necessary.

## Local reports

Full per-map outputs are ignored local research artifacts:

- `test-data/sequence-prior-remap-v4-validation-015/report.json`
- `test-data/sequence-prior-remap-v4-validation-035/report.json`
