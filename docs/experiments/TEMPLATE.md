# E-NNN — <title>

- **Status:** draft
- **Question node:** <node>
- **Hypotheses:** <hypotheses>
- **Budget:** _agent-hours_, _compute-hours_ (stop and report at 150%)
- **Proposed:** <date>
- **Approved:** — (owner, date)

## Question
One or two sentences. Why the answer matters for the node and for Q0
(see `docs/charter.md`).

## Prediction
Expected numbers, the confidence for each, and the reasoning behind them.
Written before any run and committed (registration).

## Primary metric
Exactly one metric with its aggregation (shifts, k, horizon, seeds) fixed
here. Anything else reported is exploratory.

Report: `metric_or_pattern`

## Baselines
- Floor:
- Black box:
- Known model:
- Bound / ceiling:
- Effort symmetry: which interventions each arm gets, and the tuning grids.

## Decision rule
Each plausible outcome leads to a next step. At least one branch is a
**stop**. Say why each threshold matters.

## Prerequisites
Oracle rungs or earlier experiments, with their ledger IDs.

## Plan
Commands (with `--experiment E-NNN`), seeds, expected wall-clock time.

## Result
<!-- report:begin -->
<!-- report:end -->

Reading against the prediction.

## Belief update
What changed in `claims.md` and `hypotheses.md`. Step-back paragraph:
does the node stay open, and what would the owner ask?
