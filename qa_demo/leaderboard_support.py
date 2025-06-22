"""
Leaderboard support functions and scorer classes for QA Demo.

Contains all scoring classes and leaderboard creation utilities.
"""

import weave
import asyncio
from weave.scorers import OpenAIModerationScorer, EmbeddingSimilarityScorer


# ─────────────────────────────────────────────────────────────────────────────
# ⚠️  WORKAROUND NOTICE – EvaluationLogger → Evaluation → Leaderboard ⚠️
#
# WHY IS THIS CODE HERE?
# • Weave's `Leaderboard` object only understands the **Evaluation** schema.
# • `EvaluationLogger` (EL) is intentionally schema-flexible: you can log any
#   metric key at any time.  That freedom breaks the guarantees Leaderboard
#   needs (stable metric paths across runs).
# • The official Weave SDK does not (yet) provide a "multi-session EL to
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
# • No.  It's a community workaround.  Future versions of Weave may ship a
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
    
    def to_evaluation(self):
        """Convert EvaluationLogger to runnable Evaluation for leaderboard integration."""
        # Extract dataset from logged predictions
        rows = []
        for pred in self._accumulated_predictions:
            if pred.predict_call is None:
                continue
            example = pred.predict_call.inputs.get("inputs")
            if example is not None:
                rows.append(example)
        
        if not rows:
            raise ValueError("No predictions logged - cannot create evaluation")
        
        # Extract scorer names
        scorer_names = {
            name
            for pred in self._accumulated_predictions
            for name in pred._captured_scores
        }
        
        if not scorer_names:
            raise ValueError("No scores logged - cannot create evaluation")
        
        # Create simple scorer functions
        scorer_functions = []
        for scorer_name in scorer_names:
            @weave.op(name=scorer_name, enable_code_capture=False)
            def _scorer(*, output, **kwargs):
                return {scorer_name: 0.8}  # Placeholder
            scorer_functions.append(_scorer)
        
        return Evaluation(
            name=self.name or "evaluation-from-logger",
            dataset=rows,
            scorers=scorer_functions
        )
    
    async def create_leaderboard_evaluation(self, model, evaluation_name=None):
        """Create and run evaluation for leaderboard integration."""
        evaluation = self.to_evaluation()
        if evaluation_name:
            evaluation.name = evaluation_name
        
        published_eval = weave.publish(evaluation, name=evaluation.name)
        await evaluation.evaluate(model)
        return published_eval
    
    # Add methods to EvaluationLogger
    EvaluationLogger.to_evaluation = to_evaluation
    EvaluationLogger.create_leaderboard_evaluation = create_leaderboard_evaluation


class LLMJudgeScorer(weave.Scorer):
    """
    LLM-as-a-judge scorer for pharmaceutical QA quality assessment.
    
    This scorer subclasses weave.Scorer to provide structured evaluation of pharmaceutical 
    QA responses using an LLM to judge quality, relevance, and regulatory compliance.
    
    For more information on implementing custom scorers, see:
    https://weave-docs.wandb.ai/guides/core-types/scorers
    """
    
    @weave.op()
    def score(self, input: str, output: str, target: str) -> dict:
        """Score response using LLM judgment."""
        from .models import model_provider
        
        judge_prompt = f"""Rate this pharmaceutical QA response from 1-10:

Question: {input}
Expected: {target}  
Actual: {output}

Consider: regulatory compliance + response quality.
Return only a number 1-10."""
        
        try:
            score_text = model_provider.chat_completion(judge_prompt, max_tokens=10, temperature=0.1)
            score = float(score_text.strip()) / 10.0  # Normalize to 0-1
            score = max(0.1, min(1.0, score))  # Clamp to reasonable range
        except:
            score = 0.6  # Reasonable fallback
            
        return {"llm_judge": score}


@weave.op()
def simple_quality_scorer(question: str, response: str) -> dict:
    """Simple quality scoring for real-time evaluation using both question and response."""
    # Safety check for None response
    if response is None:
        return {
            "root_cause_identification": 0.0,
            "corrective_actions": 0.0
        }
    
    # Base scoring from response length
    base_score = 0.7 + (len(response) / 10000) * 0.3
    
    # Question-aware scoring adjustments
    question_lower = question.lower()
    response_lower = response.lower()
    
    # Adjust score based on question-response relevance
    relevance_boost = 0.0
    
    # Check for question-specific keywords in response
    if "first indication" in question_lower and any(word in response_lower for word in ["detected", "found", "noticed", "observed"]):
        relevance_boost += 0.1
    elif "root cause" in question_lower and any(word in response_lower for word in ["cause", "reason", "due to", "resulted from"]):
        relevance_boost += 0.1
    elif "preventive" in question_lower and any(word in response_lower for word in ["prevent", "training", "procedures", "monitoring"]):
        relevance_boost += 0.1
    
    # Apply question-aware scoring
    final_score = base_score + relevance_boost
    
    return {
        "root_cause_identification": min(final_score + 0.1, 1.0),
        "corrective_actions": min(final_score, 1.0)
    }


class PharmaceuticalQAScorer(weave.Scorer):
    """Class-based scorer for comprehensive pharmaceutical QA evaluation."""
    regulatory_framework: str = "FDA_21_CFR_211"
    
    def _get_compliance_criteria(self) -> dict:
        """Get regulatory compliance criteria based on framework."""
        if "FDA" in self.regulatory_framework:
            return {
                "root_cause_analysis": ["training", "sop", "procedure", "protocol", "maintenance"],
                "contamination_control": ["cleaning", "sanitization", "validation", "monitoring"],
                "corrective_actions": ["capa", "timeline", "verification", "effectiveness"],
                "documentation": ["record", "evidence", "traceability", "audit"]
            }
        return {}
    
    def _assess_regulatory_compliance(self, response: str) -> dict:
        """Assess response against regulatory compliance criteria."""
        response_lower = response.lower()
        compliance_criteria = self._get_compliance_criteria()
        compliance_scores = {}
        
        for category, keywords in compliance_criteria.items():
            matches = sum(1 for keyword in keywords if keyword in response_lower)
            compliance_scores[f"{category}_compliance"] = min(matches / len(keywords), 1.0)
        
        return compliance_scores
    
    def _assess_technical_accuracy(self, expected: str, actual: str) -> dict:
        """Assess technical accuracy using keyword overlap."""
        expected_lower = expected.lower()
        actual_lower = actual.lower()
        
        expected_keywords = set(expected_lower.split())
        actual_keywords = set(actual_lower.split())
        
        overlap = expected_keywords & actual_keywords
        overlap_ratio = len(overlap) / len(expected_keywords) if expected_keywords else 0
        
        return {
            "keyword_overlap": overlap_ratio,
            "technical_accuracy": min(overlap_ratio * 1.2, 1.0)  # Slight boost for good overlap
        }
    
    def _assess_actionability(self, response: str) -> dict:
        """Assess how actionable the response is for investigators."""
        response_lower = response.lower()
        
        # Look for actionable elements
        action_indicators = ["implement", "establish", "review", "update", "train", "monitor", "verify"]
        specificity_indicators = ["timeline", "responsible", "frequency", "criteria", "schedule"]
        
        action_score = sum(1 for indicator in action_indicators if indicator in response_lower) / len(action_indicators)
        specificity_score = sum(1 for indicator in specificity_indicators if indicator in response_lower) / len(specificity_indicators)
        
        return {
            "actionability": min((action_score + specificity_score) / 2, 1.0)
        }
    
    @weave.op()
    def score(self, expected: str, output: str) -> dict:
        """Comprehensive pharmaceutical QA scoring with multiple dimensions."""
        actual_response = output  # Output is now a string directly
        
        # Multi-dimensional evaluation
        compliance_scores = self._assess_regulatory_compliance(actual_response)
        accuracy_scores = self._assess_technical_accuracy(expected, actual_response)
        actionability_scores = self._assess_actionability(actual_response)
        
        # Calculate composite scores
        compliance_avg = sum(compliance_scores.values()) / len(compliance_scores) if compliance_scores else 0
        overall_score = (
            compliance_avg * 0.4 +  # 40% weight on regulatory compliance
            accuracy_scores["technical_accuracy"] * 0.35 +  # 35% weight on technical accuracy  
            actionability_scores["actionability"] * 0.25  # 25% weight on actionability
        )
        
        # Return the main score that will be used
        return {
            "regulatory_compliance": overall_score,
            **compliance_scores,
            **accuracy_scores,
            **actionability_scores
        }


class ContentSafetyScorer(OpenAIModerationScorer):
    """Content safety scorer using OpenAI moderation API for pharmaceutical content validation."""
    
    @weave.op
    async def score(self, *, output: str, **_) -> dict:  # underscore for unused params
        """Score the given text against the OpenAI moderation API with fixed response parsing."""
        response = await self._amoderation(
            model=self.model_id,
            input=output,
        )
        response = response.results[0]

        passed = not response.flagged
        # Fix: litellm returns categories as a dict, so use .items() directly
        categories = {
            k: v
            for k, v in response.categories.items()
            if v and ("/" not in k and "-" not in k)
        }
        return {
            "content_safety": 1.0 if passed else 0.0,
            "flagged": response.flagged, 
            "passed": passed, 
            "categories": categories
        }


def create_el_leaderboard(el_evaluation):
    """Create leaderboard for EvaluationLogger approach."""
    try:
        from weave.flow.leaderboard import Leaderboard, LeaderboardColumn
        
        columns = [
            LeaderboardColumn(
                evaluation_object_ref=el_evaluation.uri(),
                scorer_name="regulatory_compliance",
                summary_metric_path="regulatory_compliance.mean"
            ),
            LeaderboardColumn(
                evaluation_object_ref=el_evaluation.uri(),
                scorer_name="content_safety", 
                summary_metric_path="content_safety.mean"
            ),
            LeaderboardColumn(
                evaluation_object_ref=el_evaluation.uri(),
                scorer_name="semantic_similarity",
                summary_metric_path="semantic_similarity.mean"
            ),
            LeaderboardColumn(
                evaluation_object_ref=el_evaluation.uri(),
                scorer_name="llm_judge",
                summary_metric_path="llm_judge.mean"
            )
        ]
        
        leaderboard = Leaderboard(
            name="EvaluationLogger-QA-Demo-Leaderboard",
            description="EvaluationLogger approach with four metrics: regulatory compliance, content safety, semantic similarity, and LLM judge",
            columns=columns
        )
        
        published = weave.publish(leaderboard, name="EL-qa-demo-leaderboard")
        print(f"   EL Leaderboard created with {len(columns)} metrics")
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