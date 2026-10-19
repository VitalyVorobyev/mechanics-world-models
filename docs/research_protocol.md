# Research Protocol (v2, 2026-10-19)

How this project decides what to run, how it records results and how it
grows a verified set of facts. The protocol is binding for humans and
agents. v1 (2026-10-18) introduced predictions, decision rules and the
oracle ladder. v2 makes them **verifiable** and adds rules against
circling. Why: [`review-2026-10-19.md`](review-2026-10-19.md), Part B.

## 1. The documents and what each is for

| File | Answers | Size rule |
|---|---|---|
| [`charter.md`](charter.md) | Why, the root question, the question tree, budget, stop date, decision log | ≤ 1 page; changes only by owner decision |
| [`claims.md`](claims.md) | What we know, graded A/B/C with ledger evidence | one short entry per claim |
| [`current_status.md`](current_status.md) | What is happening now: active node, approved/running experiments, last decisions | ≤ 60 lines; **rewritten**, not appended |
| [`hypotheses.md`](hypotheses.md) | Index of hypotheses and assumptions, one line each | one table row per entry |
| `experiments/E-NNN-*.md` | One experiment: prediction, decision rule, result, belief update | append-only after registration |
| [`../results/ledger.jsonl`](../results/ledger.jsonl) | Every run: commit, tree state, experiment, seed, verdict, file hash | machine-written |

History goes to `archive/` and `experiments/archive/`, never into the top-level files.

## 2. Evidence

- **A result counts only if it comes from committed code.** Oracle CLIs call
  `research.provenance.begin_run`, which refuses a dirty `src/` or
  `scripts/` tree. `--allow-dirty` exists for exploration, and its result
  is ungradeable.
- Each record stores:
  - the full commit and the dirty paths;
  - argv and package versions;
  - a data fingerprint and a source fingerprint.

  Each run appends one line to the committed ledger, with the sha256 of
  its result file. Caches are keyed by the source fingerprint.
- **Tables are generated, never typed:** `scripts/research/report.py`.
  It reports the sample std over seeds and marks superseded reruns.
- **No scratch evidence.** If an analysis decides something, commit its
  script (under `scripts/oracles/` or `scripts/research/`) and run it from
  a clean tree. A number from a scratch script is grade C at best.

## 3. Experiment lifecycle

```
propose → register (commit) → approve (owner) → run → report → claims → step-back
```

1. **Propose.** Run `scripts/research/new_experiment.py --slug … --node Q…`.
   The record names a charter node. If no node fits, it needs a charter
   change, which is an owner decision.
2. **Register.** Fill in the protected sections and commit the record:
   - **Prediction:** numbers with confidence, and the reasoning.
   - **Primary metric:** exactly one, with its aggregation (shifts, k,
     horizon, seeds) fixed in advance, plus a `Report:` line listing the
     metric patterns.
   - **Baselines:** the bracket (§5) and the effort-symmetry note (§6).
   - **Decision rule:** every plausible outcome maps to a next step. At
     least one branch says **stop** (drop the node, or write it up). Each
     threshold has a reason, an effect size that matters, not a round
     number.
   
   Also add a budget (agent-hours, compute-hours) and the prerequisites
   (oracle rungs, with their latest ledger IDs). Set the status to
   `registered`.
3. **Approve.** The owner sets the status to `approved`, or tells the
   agent to. Runs that are minutes long and only read existing results
   need no approval. Anything that trains, simulates at scale or takes
   more than about 10 minutes does.
4. **Run** with `--experiment E-NNN` from a clean tree. The check
   (`scripts/research/check.py`, also in the test suite) fails if:
   - the record was not committed before the run's commit;
   - a protected section changed after a run;
   - the record was not approved.
5. **Report.** Run `scripts/research/report.py E-NNN`. It writes the table
   into the record and `results/E-NNN/summary.json`. Below it, write the
   reading against the prediction: confirmed, refuted, or not as
   predicted, and why.
6. **Claims.** Add, upgrade, narrow or supersede entries in `claims.md`,
   and update the index row in `hypotheses.md`.
7. **Step back.** One paragraph in the record:
   - Did this change what we believe about Q0?
   - Does the node stay open?
   - What would the owner most likely ask?

## 4. Rules against circling

- **Chain-depth limit.** Three consecutive experiments on one node that
  each end in "diagnose further" trigger a **zoom-out note**:
  - Is this node still on the path to Q0?
  - What is the value of the next answer?
  - What is the cheapest way to make the node moot?

  Then the owner decides. No fourth diagnostic without that decision.
- **Budgets are real.** At 150% of an experiment's budget, stop and report.
- **The stop date in the charter is a decision point, not a deadline to
  stretch.**
- **Each decision rule needs a stop branch.** "Otherwise, look at X" alone
  is not a decision rule.
- **Write before running.** If no outcome would change the next step, do
  not run.

## 5. Bracketing: floors, ceilings, known models

Never report a number without:

- **Floor:** a trivial predictor. That is copy-last-frame or the mean
  for pixels, and no adaptation (k = 0) for dynamics.
- **Black box:** the unstructured model trained on the same data and
  conditions.
- **Known model:** wherever the physics is known (cartpole), the known
  equations with fitted parameters. A learned structure is a contribution
  only if it is compared with the model it imitates.
- **Bound:** the oracle ceiling (true state, true parameters) and, for
  parameter estimates, the Cramér–Rao bound under the stated noise model.

## 6. Effort symmetry

When an intervention is designed to help one arm, such as a gauge fix, a
robust loss, a new slot or a better start state, either:
- offer the matching intervention to the other arms, or
- record the asymmetry in the record's Baselines section.

Report each arm's tuning budget (the grid sizes). Selection protocols are
identical across arms.

## 7. Surprises (strong inference)

1. List **at least three** competing explanations, including "the
   measurement is wrong".
2. Pick the cheapest test that tells them apart. Usually that is a probe,
   an oracle swap or a simulation, not a new training run.
3. Record the eliminated explanations in the record.

## 8. The oracle ladder

The simulator gives ground truth, so test each component against the
truth before end-to-end training. Climb in order. If a rung fails, fix
that component. Tools live in `scripts/oracles/`, and results go to
`runs/oracles/` and the ledger.

| Rung | Question | Tool |
|---|---|---|
| 0 | Does the manipulation change what we think, by enough to measure? | `intervention_effect.py` |
| 1 | Can the decoder or renderer draw frames from the true state? | `decoder_oracle.py` |
| 2 | Can the encoder or front end read the true state, by visibility and pose? | `encoder_oracle.py`, `bootstrap_oracle.py`, `frozen_renderer_oracle.py` |
| 3 | On true state, how do the dynamics classes compare? What does the true dynamics achieve? | `dynamics_oracle.py`, `dynamics_ceiling.py`, `mass_gauge.py`, `context_oracle.py` |
| 3n/3p | Do the conclusions survive simulated noise, then real front-end poses? | `context_oracle.py --pose-noise / --encoder / --front-end` |
| 4–6 | Overfit one batch, pilot, full run | trainers (frozen with the pixel world models) |

## 9. Statistics

- Estimate the expected effect size before the run, from rung 0 or rung 3.
- Use at least 3 seeds before claiming a difference between arms. Report
  the sample std and the per-seed values. A difference whose sign is not
  consistent across seeds is not a difference.
- Split by episode for any probe or held-out metric.
- Fix the primary metric and its aggregation before running. Any other
  metric reported is exploratory.

## 10. Reviews and literature

- **At each decision point** (a node closes, the stop date, a pivot), a
  fresh agent with no context gets [`review_prompt.md`](review_prompt.md)
  and the top-level docs. Its findings go into the decision log or the
  claims.
- **Before any claim of novelty, and before registering an experiment
  that targets a new node**, run a targeted literature check: has this
  been answered, and how did others do it? Note the result in the record.

## 11. Agent session routine

- **Start:** read [`charter.md`](charter.md), [`current_status.md`](current_status.md),
  [`claims.md`](claims.md) and the open experiment records. Run
  `scripts/research/check.py`.
- **During:** work only on approved experiments or on proposals for them.
  Commit before running.
- **End:**
  - generate reports;
  - update the claims and the index;
  - **rewrite** `current_status.md`;
  - run the check;
  - list for the owner the decisions that are waiting on them.

## 12. Housekeeping

- Pin the environment (`requirements.lock.txt`).
- One commit per coherent change. The record's registration is its own
  commit, made before the run.
- `runs/` is gitignored. The ledger and `results/E-NNN/summary.json` are
  committed.
