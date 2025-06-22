"""
Leaderboard support functions for QA Demo.

This module provides:
1. Bridge functionality to convert EvaluationLogger sessions to Evaluation objects
2. Leaderboard creation utilities for both EvaluationLogger and standard Evaluation workflows

── STATIC EVALUATION NOTICE ─────────────────────────────────────────────
The bridge now publishes a synthetic `weave.Evaluation` **without** calling
`evaluation.evaluate()`.  All scalar scores are injected and aggregate means
are pre-computed.  As a result:
  • No model re-runs occur (zero extra cost).
  • Leaderboards can display metrics immediately.
  • Calling `evaluation.evaluate()` later will produce stale data.

Bridge Functions:
- add_leaderboard_support(): Patches EvaluationLogger with leaderboard integration methods
- EvaluationLogger.to_evaluation(): Converts EL session to synthetic Evaluation
- EvaluationLogger.create_leaderboard_evaluation(): Convenience wrapper for leaderboard creation

Leaderboard Builders:
- create_el_leaderboard(): Creates leaderboard from EvaluationLogger-based evaluations
- create_standard_leaderboard(): Creates leaderboard from standard Evaluation objects

For scorer implementations, see scorers.py.
"""

import weave
import asyncio


# ─────────────────────────────────────────────────────────────────────────────
# ⚠️  WORKAROUND NOTICE – EvaluationLogger → Evaluation → Leaderboard ⚠️
#
# WHY IS THIS CODE HERE?
# • Weave's `Leaderboard` object only understands the **Evaluation** schema.
# • `EvaluationLogger` (EL) is intentionally schema-flexible: you can log any
#   metric key at any time.  That freedom breaks the guarantees Leaderboard
#   needs (stable metric paths across runs).
# • The Weave SDK does not provide a "multi-session EL to
#   leaderboard" helper.  Hence this bespoke bridge.
#
# WHAT DOES IT DO?
# • `add_leaderboard_support()` monkey-patches two helper methods onto
#   `EvaluationLogger`.
#   1. `to_evaluation()` – rewrites one EL session into a *synthetic*
#      `weave.Evaluation`:  
#        – Each logged prediction row is copied.  
#        – All scalar scores logged via `log_score()` are **injected** back
#          into that row so a trivial scorer can surface them.  
#        – A lightweight scorer is generated per metric key so the evaluation
#          has a stable path:  <metric_key>.mean
#   2. `create_leaderboard_evaluation()` – convenience wrapper that publishes
#      and (optionally) re-runs the evaluation so it becomes leaderboard-ready.
#
# HARD CONSTRAINTS / PREREQUISITES
# • Every EL session **must log the exact same metric keys** or the resulting
#   leaderboard columns will be sparse.  We enforce nothing here—discipline is
#   on the user.
# • This bridge only captures **scalar** scores.  Complex/nested scorer output
#   will be flattened down to a single mean per metric.
# • Calling this helper per EL session means you'll end up with one evaluation
#   object per model variant; those are then aggregated manually into a
#   leaderboard elsewhere in the code.
#
# IS THIS OFFICIALLY SUPPORTED?
# • No.  It's a workaround.  Future versions of Weave may ship a
#   first-class solution that renders this obsolete—or incompatible.
#
# SHOULD YOU USE IT?
# • Use it if you value EL's flexible, incremental logging but *also* need the
#   side-by-side comparison that Leaderboards provide.  
# • Avoid it for long-term, mission-critical benchmarking workflows; a native
#   `Evaluation` run is simpler, more reproducible, and better-supported.
#
# REVIEWER TIP
# • If you see blanks in the EL-based leaderboard, first verify that every
#   EL session logged the same metric keys and that the key names match the
#   leaderboard column definitions 1-for-1.
# ─────────────────────────────────────────────────────────────────────────────

def add_leaderboard_support():
    """Add leaderboard integration methods to EvaluationLogger."""
    from weave.flow.eval_imperative import EvaluationLogger
    from weave.flow.eval import Evaluation
    import copy
    
    def to_evaluation(self):
        """Convert EvaluationLogger to **static** Evaluation for leaderboard integration (no re-run)."""
        # Collect rows and associated scores
        rows = []
        all_metric_names = set()
        prediction_scores: dict[int, dict[str, float]] = {}
        
        for i, pred in enumerate(self._accumulated_predictions):
            if pred.predict_call is None:
                continue
                
            # Deep-copy each pred.predict_call.inputs["inputs"] into row
            example = pred.predict_call.inputs.get("inputs")
            if example is not None:
                rows.append(copy.deepcopy(example))
                
                # Store scores associated with this prediction
                prediction_scores[i] = pred._captured_scores
                all_metric_names.update(pred._captured_scores.keys())
        
        if not rows:
            raise ValueError("No predictions logged - cannot create evaluation")
        
        if not all_metric_names:
            raise ValueError("No scores logged - cannot create evaluation")
        
        # Generate one lightweight scorer per metric_name that returns the captured score
        scorer_functions = []
        prediction_scores_list = [prediction_scores[i] for i in sorted(prediction_scores.keys())]

        for metric_name in all_metric_names:
            def make_scorer(metric_name, prediction_scores_seq):
                # Use concise op names (just the metric key)
                @weave.op(name=f"{metric_name}", enable_code_capture=False)
                def _scorer(output, **kwargs):
                    # Maintain per-scorer call index to align with dataset row order
                    if not hasattr(_scorer, "_call_idx"):
                        _scorer._call_idx = 0  # type: ignore[attr-defined]

                    idx = _scorer._call_idx  # type: ignore[attr-defined]
                    _scorer._call_idx += 1  # type: ignore[attr-defined]

                    if idx < len(prediction_scores_seq):
                        return {metric_name: prediction_scores_seq[idx].get(metric_name, 0.0)}
                    return {metric_name: 0.0}
                return _scorer

            scorer_functions.append(make_scorer(metric_name, prediction_scores_list))
        
        evaluation = Evaluation(
            name=self.name or "evaluation-from-logger",
            dataset=rows,
            scorers=scorer_functions,
        )

        return evaluation
    
    async def create_leaderboard_evaluation(self, evaluation_name: str | None = None):
        """Publish Evaluation and run a replay model so Leaderboard paths materialise."""
        evaluation = self.to_evaluation()
        if evaluation_name:
            evaluation.name = evaluation_name

        # ── Build captured outputs so the replay model can surface them during evaluation ──
        captured_outputs: list[str | None] = []
        for i, pred in enumerate(self._accumulated_predictions):
            if pred.predict_call is not None:
                captured_outputs.append(pred.predict_call.output)

        # ── Unique replay op per evaluation to create distinct model alias ──
        safe_name = evaluation.name.replace(" ", "_").replace("-", "_")
        op_name = safe_name if safe_name.startswith("replay_") else f"replay_{safe_name}"

        @weave.op(name=op_name, enable_code_capture=False)  # type: ignore[arg-type]
        def _replay_model(**kwargs):  # noqa: ANN001
            """Return the captured output for the current dataset row (best-effort order)."""
            # Simple counter-based mapping: Weave calls the model once per row in order.
            if not hasattr(_replay_model, "_call_count"):
                _replay_model._call_count = 0  # type: ignore[attr-defined]

            idx = _replay_model._call_count  # type: ignore[attr-defined]
            _replay_model._call_count += 1  # type: ignore[attr-defined]

            if 0 <= idx < len(captured_outputs):
                return captured_outputs[idx]
            return None

        # Run evaluation so <metric>.mean attributes materialise for Leaderboard
        await evaluation.evaluate(model=_replay_model, __weave={"display_name": evaluation.name})
        
        # Publish *after* evaluation so URI version includes results
        published_eval = weave.publish(evaluation, name=evaluation.name)

        return published_eval
    
    # Patch methods onto EvaluationLogger
    EvaluationLogger.to_evaluation = to_evaluation
    EvaluationLogger.create_leaderboard_evaluation = create_leaderboard_evaluation



def create_el_leaderboard(evaluation_objects):
    """Create leaderboard for EvaluationLogger approach."""
    try:
        from weave.flow.leaderboard import Leaderboard, LeaderboardColumn
        
        # Accept list evaluation_objects
        if not isinstance(evaluation_objects, list):
            evaluation_objects = [evaluation_objects]
        
        # For each object, add four LeaderboardColumn instances
        columns = []
        for el_evaluation in evaluation_objects:
            for metric_key in [
                "regulatory_compliance",
                "content_safety",
                "semantic_similarity",
                "llm_judge",
            ]:
                columns.append(
                    LeaderboardColumn(
                        evaluation_object_ref=el_evaluation.uri(),
                        scorer_name=f"{metric_key}",
                        summary_metric_path=f"{metric_key}.mean",
                    )
                )
        
        leaderboard = Leaderboard(
            name="EvaluationLogger-QA-Demo-Leaderboard",
            description=f"EvaluationLogger approach comparing {len(evaluation_objects)} model variants across four metrics: regulatory compliance, content safety, semantic similarity, and LLM judge",
            columns=columns
        )
        
        published = weave.publish(leaderboard, name="EL-qa-demo-leaderboard")
        print(f"   EL Leaderboard created with {len(evaluation_objects)} models × 4 metrics = {len(columns)} columns")
        return published
        
    except Exception as e:
        print(f"   EL Leaderboard creation failed: {e}")
        return None


def create_standard_leaderboard(evaluation_objects):
    """Create leaderboard for standard evaluation approach with all model variants."""
    try:
        from weave.flow.leaderboard import Leaderboard, LeaderboardColumn
        
        # Create columns for all evaluations (all 4 model variants)
        columns = []
        for eval_obj in evaluation_objects:
            # Create 4 columns per evaluation (one for each metric)
            columns.extend([
                LeaderboardColumn(
                    evaluation_object_ref=eval_obj.uri(),
                    scorer_name="PharmaceuticalQAScorer",
                    summary_metric_path="regulatory_compliance.mean"
                ),
                LeaderboardColumn(
                    evaluation_object_ref=eval_obj.uri(),
                    scorer_name="ContentSafetyScorer",
                    summary_metric_path="content_safety.mean"
                ),
                LeaderboardColumn(
                    evaluation_object_ref=eval_obj.uri(),
                    scorer_name="EmbeddingSimilarityScorer", 
                    summary_metric_path="similarity_score.mean"
                ),
                LeaderboardColumn(
                    evaluation_object_ref=eval_obj.uri(),
                    scorer_name="LLMJudgeScorer",
                    summary_metric_path="llm_judge.mean"
                )
            ])
        
        leaderboard = Leaderboard(
            name="Standard-Evaluation-QA-Demo-Leaderboard", 
            description=f"Standard Evaluation approach comparing {len(evaluation_objects)} model variants across four metrics: regulatory compliance, content safety, semantic similarity, and LLM judge",
            columns=columns
        )
        
        published = weave.publish(leaderboard, name="EVAL-qa-demo-leaderboard")
        print(f"   Standard Leaderboard created with {len(evaluation_objects)} models × 4 metrics = {len(columns)} columns")
        return published
        
    except Exception as e:
        print(f"   Standard Leaderboard creation failed: {e}")
        return None 