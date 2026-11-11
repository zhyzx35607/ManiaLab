# LN-Transition Hard-Negative Probe

Updated: 2026-11-11

## Purpose

The current sequence prior treats the authored lane assignment as positive and
compares it with randomly generated feasible lane changes. This probe measures
whether a separate training objective can generate harder alternatives that
change an LN interaction while keeping every note's onset and end fixed.

For LN-context windows, the new policy changes one event row at a time and
keeps an alternative only when its interaction counts differ in at least one
of these categories: inverse, reverse shield, shield, same-hand hybrid,
split release, release-press, or nearby release-press. It still requires the
full local path to remain feasible. Tap-only windows keep the existing random
negative policy.

The authored pattern remains the positive label. A changed category is not a
judgment that the alternative is worse or less fun; it only makes a more
specific ranking comparison available. This probe does not train or score a
model, select note events, or alter LN lengths.

## 10K Validation Coverage

The fixed validation split contains 45 maps from 13 song groups. With 80
sampled windows per map and two negatives per retained window:

| Measure | Random feasible | LN-transition hard |
|---|---:|---:|
| Sampled LN-context windows | 1,461 | 1,461 |
| Retained LN-context windows | 1,461 | 1,360 |
| LN windows unable to produce two required negatives | 0 | 101 |
| Found negative examples | 7,196 total | 6,999 total |

The hard policy retained 93.1% of sampled LN-context windows. It failed to
produce the full pair for 101 windows across 12 of the 13 groups; the
remaining group had no sampled LN-context windows. Across accepted hard
examples, category changes were counted 1,092 times for same-hand hybrid,
1,541 for release-press, 98 for inverse, 92 for reverse shield, 79 for
shield, 92 for split release, and 107 for nearby release-press. One example
can change more than one category.

The detailed report is kept locally at
`test-data/sequence-ln-hard-negative-probe-v4/report.json`. The reproducible
entry point is `scripts/probe_sequence_ln_negatives.py`. The probe reads only
10K validation charts and does not read the independent test split's chart
contents.

## Decision

Coverage is sufficient to run an isolated training experiment, with the
expected loss of some sparse or hard-to-perturb LN windows. The experiment
must use the new `sequence-window-cache-v4-ln-transition` cache and compare
LN-context and tap-only ranking by independent song group. The original
sequence prior and the application's default behavior remain unchanged.

Any ranking result remains a model-behavior check, not a quality verdict.
Human review and actual play testing are still required; learning LN duration
requires separate audio-grounded release evidence.
