# QA Demo – Option B & File-Split Refactor Checklist

This document guides an AI coding assistant through the **EvaluationLogger → Leaderboard** alignment refactor *and* the scorer / leaderboard file split.  Every task has a checkbox – replace `[ ]` with `[x]` when complete.

---

## 1  EvaluationLogger → Evaluation bridge (Option B implementation)

### 1.1  Patch `add_leaderboard_support.to_evaluation`

- [ ] Deep-copy each `pred.predict_call.inputs["inputs"]` into `row`.
- [ ] Inject every scalar score from `pred._captured_scores` into that `row`.
- [ ] Collect rows into a `rows` list.
- [ ] Gather the set of all `metric_names` seen.
- [ ] Generate one lightweight scorer *per* `metric_name` that simply returns the injected value.
- [ ] Return a `weave.Evaluation` built from `rows` and the generated scorers.

### 1.2  Update `create_leaderboard_evaluation`

- [ ] Call `evaluation.evaluate(model=None)` (avoids expensive re-runs).
- [ ] Publish & return the evaluation as before.

---

## 2  Part 2 workflow updates (multi-evaluation conversion)

- [ ] After EL loop, iterate over **all** `all_el_sessions`:
  - [ ] Convert each EL session via `create_leaderboard_evaluation(model=None, evaluation_name=f"EL_eval_{model_name}")`.
  - [ ] Append each returned object to `evaluation_objects`.
- [ ] Adapt call signature of `create_el_leaderboard(evaluation_objects)` (see §3).

---

## 3  Leaderboard helper refactor

### 3.1  `create_el_leaderboard`

- [ ] Accept **list** `evaluation_objects`.
- [ ] For each object, add four `LeaderboardColumn` instances (one per metric: `regulatory_compliance`, `content_safety`, `semantic_similarity`, `llm_judge`).

### 3.2  `create_standard_leaderboard`

- [ ] Confirm still works with list input (no change needed).

---

## 4  Dataset clean-up (remove `expected` field)

- [ ] Delete the `"expected": …` key from every dataset entry in `part2_evaluation_logger.py`.
- [ ] Delete the same key from every dataset entry in `part3_evaluation.py`.
- [ ] Update surrounding comments accordingly.

---

## 5  File split – decouple scorers from leaderboard helpers

### 5.1  Create `qa_demo/scorers.py`

- [ ] Move the following from `leaderboard_support.py` into this file **without modification**:
  - `PharmaceuticalQAScorer`
  - `ContentSafetyScorer`
  - `LLMJudgeScorer`
  - `simple_quality_scorer`

### 5.2  Trim `qa_demo/leaderboard_support.py`

- [ ] Leave only:
  - `add_leaderboard_support()` (bridge logic)
  - `create_el_leaderboard()`
  - `create_standard_leaderboard()`
- [ ] Insert the WORKAROUND NOTICE banner describing the bridge & caveats.

### 5.3  Update imports

- [ ] In **all** modules, replace `from .leaderboard_support import …Scorer…` with `from .scorers import …Scorer…`.

### 5.4  Add file-level doc-strings

- [ ] `scorers.py` → explain purpose and state "No leaderboard code belongs here".
- [ ] `leaderboard_support.py` → sectioned doc-string distinguishing bridge vs. builders.

---

## 6  House-keeping & verification

- [ ] Run `uv run python -m qa_demo.main --mode eval_logger` → confirm EL leaderboard shows 4 rows × 4 metrics populated.
- [ ] Run `uv run python -m qa_demo.main --mode evaluation` → confirm standard leaderboard unaffected.
- [ ] Lint: `uv run ruff .` (or equivalent) passes.

---

## 7  Update master playbook

- [ ] Add a new checkbox line to root-level `refactoring.md`:

```
- [x] Extract scorers into scorers.py; clarify leaderboard_support.py roles
```

Commit & push when all boxes above are `[x]`. 