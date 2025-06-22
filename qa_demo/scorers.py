"""
Scorer classes for QA Demo evaluation.

This module contains all scorer implementations used for evaluating pharmaceutical QA responses.
These scorers assess various dimensions including regulatory compliance, content safety, 
semantic similarity, and LLM-based judgment.

No leaderboard code belongs here - this module is purely focused on scoring functionality.
For leaderboard creation and evaluation bridges, see leaderboard_support.py.
"""

import weave
import asyncio
from weave.scorers import OpenAIModerationScorer, EmbeddingSimilarityScorer


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
    
    def _assess_technical_accuracy(self, target: str, actual: str) -> dict:
        """Assess technical accuracy using keyword overlap."""
        expected_lower = target.lower()
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
    def score(self, target: str, output: str) -> dict:
        """Comprehensive pharmaceutical QA scoring with multiple dimensions."""
        actual_response = output  # Output is now a string directly
        
        # Multi-dimensional evaluation
        compliance_scores = self._assess_regulatory_compliance(actual_response)
        accuracy_scores = self._assess_technical_accuracy(target, actual_response)
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