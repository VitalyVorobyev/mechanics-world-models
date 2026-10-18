# Research Protocol

How this project decides what to run, how it reads results, and how it
keeps track of what it believes. Adopted 2026-10-18 after the review in
[`current_status.md`](current_status.md). Almost every expensive mistake in
April 2026 would have been caught by one of the cheap checks below. The
literature confirmed those mistakes, but we did not need it to find them.

The protocol is binding for humans and agents alike (see `AGENTS.md` /
`CLAUDE.md`). Keep it short; change it when it stops paying for itself.

## 1. The hypothesis registry is the source of truth

[`hypotheses.md`](hypotheses.md) lists every hypothesis **and every
assumption** the work relies on. "Assumption" includes things like "the
decoder can render the pose" or "this perturbation changes the dynamics".
Each entry has:

- an ID,
- the statement,
- a status (`assumed`, `open`, `supported`, `refuted`, `superseded`),
- the evidence (file paths, metric values, citations),
- the cheapest test that would falsify it.

Narrative docs (`current_status.md`, model docs) cite registry IDs instead
of restating beliefs. An interpretation that is not in the registry is not
yet a finding.

## 2. Before any run longer than about an hour

Write a short block in the registry entry the run is meant to test:

1. **Prediction:** the expected numbers and the reasoning behind them.
2. **Decision rule:** what each plausible outcome changes about the next
   step. If no outcome would change anything, do not run it.
3. **Prerequisites:** the oracle rungs (section 3) this run depends on, and
   their latest results.
4. **Cost:** wall-clock time, and whether the run can be stopped and
   resumed.

## 3. The oracle ladder

The simulator gives ground truth (`physics_qpos` / `physics_qvel`, physics
parameters), so every component can be tested against the truth in
isolation before end-to-end training. Climb in order; each rung is minutes
to an hour. Tools live in `scripts/oracles/`, and results go to
`runs/oracles/<name>/<timestamp>.json` with pass/fail against declared
criteria.

| Rung | Question | Tool | Typical cost |
|---|---|---|---|
| 0. Intervention | Does the manipulation change what we think, by enough to measure? | `intervention_effect.py` | 1 min |
| 1. Decoder oracle | Can the decoder render frames from the *true* state? | `decoder_oracle.py` | 10–30 min |
| 2. Encoder oracle | Can the encoder regress the *true* state from pixels? | `encoder_oracle.py` | 10–30 min |
| 3. Dynamics oracle | On *true* state trajectories, how do the dynamics classes compare (fit, rollout, adaptation)? | `dynamics_oracle.py` | minutes on CPU |
| 4. Overfit one batch | Does the full model drive its loss down on 16 sequences? | `overfit_batch.py` | 5–10 min |
| 5. Pilot | Short real run with live probes: does the signal of interest move? | trainer with `--epochs 1` | ≈ 1 h |
| 6. Full run | Confirm, with seeds | `gate_pipeline.sh`-style resumable runner | hours |

A full run is for confirming, not exploring. If a rung fails, stop climbing
and fix that component.

## 4. Floors and ceilings for every metric

Never report a number without the trivial floor and the oracle ceiling next
to it:

- **Floor:** copy the last context frame, or predict the mean (both
  evaluators already report `copy_last_frame_mse_h*`).
- **Ceiling:** the same model class run on the true state (rung 3), or the
  decoder oracle's reconstruction error (rung 1).

## 5. When something surprising happens

1. List **at least three** competing explanations, including "the
   measurement is wrong".
2. Pick the cheapest test that separates them (strong inference). Usually
   that is a probe, an oracle swap or a simulation, not a new training run.
3. Record the outcome in the registry, including the explanations that
   were eliminated.

Worked example (April → October 2026). Observation: the latent `q` had a
near-zero linear probe onto the physical state. Candidates:

- gauge freedom;
- posterior collapse;
- pose stored in `z` (bypass);
- a probe that cannot work with a raw, unwrapped angle.

A probe on `z` plus an MLP probe with an angle-aware target separated them
in two minutes: bypass, plus a broken probe. In April a single explanation
was written down and a day of experiments planned around it.

## 6. Instrumentation is part of the model

- Log what tells the competing explanations apart, not only the loss:
  information routing (probes per latent block), floors, effect sizes.
- Long runs must be observable (`progress.json`, live probe) and
  resumable (mid-epoch checkpoints, graceful stop).
- Metrics must be comparable across models on identical sequences, using
  the same masks and windows.

## 7. Statistics

- Estimate the expected effect size before the run, from rung 0 or 3. Pole
  mass ×2 moves cartpole trajectories by about 0.002 over 20 steps; no
  pixel model will resolve that.
- Use at least 3 seeds before claiming a difference between arms; report
  the spread.
- Split by episode for any probe or held-out metric.

## 8. Review cadence

- **At each milestone** (a gate passed or failed, or a pivot): run an
  adversarial review of the current story. Use a fresh agent with no
  context, or an external model or colleague, asked "what is the most
  likely reason this is wrong?". File the findings in the registry.
- **Before any long run:** check the literature specifically for whether
  this question has already been answered, and how others made the method
  work.
- **Monthly, or when resuming after a pause:** re-test the registry
  entries still marked `assumed` that the current plan depends on.

## 9. Housekeeping

- Pin the environment (`requirements.lock.txt`). Record simulator versions
  and `frame_dt` in data.
- One commit per coherent change. Registry updates travel with the code or
  results that justify them.
- Scratch experiments go in `runs/` (gitignored); anything that decides
  something is summarized in the registry with its path.
