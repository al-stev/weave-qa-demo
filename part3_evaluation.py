"""
Part 3 – Evaluations ("rigorous evaluations of AI applications")

Learning Objective:
Demonstrate the formal, batch-style weave.Evaluation workflow for systematic
comparison of multiple model variants using the same four metrics as Part 2.

This showcases the structured evaluation approach for comprehensive model comparison.
"""

import weave
import asyncio
from weave.scorers import EmbeddingSimilarityScorer

from models import create_model_variants, initialize_model_provider
from scorers import (
    PharmaceuticalQAScorer,
    ContentSafetyScorer, 
    LLMJudgeScorer
)
from leaderboard_support import (
    create_standard_leaderboard
)


def part3_evaluation():
    """
    Part 3 – Rigorous evaluation using weave.Evaluation for systematic model comparison.
    
    Uses the same four metrics as Part 2 to enable apples-to-apples comparison between
    EvaluationLogger and standard Evaluation workflows.
    """
    
    print("\n" + "="*60)
    print(" Part 3 – Evaluations")
    print("="*60)
    print("Goal: Rigorous evaluations of AI applications")
    
    # Initialize model provider
    if not initialize_model_provider():
        print("Failed to initialize model provider")
        return
    
    # Create model variants
    model_variants = create_model_variants()
    
    # Create comprehensive evaluation dataset
    evaluation_dataset = [
        {
            "input": "What was the first indication that contamination occurred?",
            "target": "Quality control testing detected unexpected impurities during routine batch analysis."
        },
        {
            "input": "What was the root cause of the contamination?", 
            "target": "Inadequate environmental monitoring, compromised HEPA filtration, and insufficient personnel training protocols."
        },
        {
            "input": "What CAPA plan will prevent recurrence of cross-contamination?",
            "target": "Enhanced facility segregation, updated cleaning validation protocols, and comprehensive staff retraining with competency assessment."
        },
        {
            "input": "How should we document this investigation for regulatory submission?",
            "target": "Complete investigation report with timeline, evidence, root cause analysis, CAPA plan, and effectiveness verification."
        },
        {
            "input": "What immediate containment actions are required?",
            "target": "Halt production, quarantine affected batches, conduct risk assessment, and notify quality assurance management immediately."
        }
    ]
    
    # Publish standardized evaluation dataset
    eval_dataset = weave.Dataset(name="QA-demo-evaluation-dataset", rows=evaluation_dataset)
    weave.publish(eval_dataset, name="QA-demo-evaluation-dataset")
    print(f" Published evaluation dataset with {len(evaluation_dataset)} scenarios")
    
    # Create the same four scorers as Part 2 for consistent comparison
    comprehensive_scorers = [
        PharmaceuticalQAScorer(regulatory_framework="FDA_21_CFR_211"),
        ContentSafetyScorer(),
        EmbeddingSimilarityScorer(),
        LLMJudgeScorer()
    ]
    
    print(f"\n Step 1: Systematic multi-model evaluation")
    print(f"   Evaluating {len(model_variants)} model variants with {len(comprehensive_scorers)} scorers")
    
    # Store evaluation results for leaderboard
    evaluation_results = []
    evaluation_objects = []
    
    for model_name, model in model_variants.items():
        print(f"\n   → Evaluating {model_name}...")
        
        # Create evaluation with same four metrics as Part 2
        evaluation = weave.Evaluation(
            name=f"QA_Demo_{model_name}",
            dataset=evaluation_dataset,
            scorers=comprehensive_scorers
        )
        
        # Publish evaluation
        published_eval = weave.publish(evaluation, name=f"QA_Demo_{model_name}")
        evaluation_objects.append(published_eval)
        
        # Run evaluation
        result = asyncio.run(evaluation.evaluate(model))
        evaluation_results.append((model_name, result))
        
        print(f"     {model_name} evaluation complete")
        print(f"       Regulatory compliance: {result.get('PharmaceuticalQAScorer', {}).get('regulatory_compliance', {}).get('mean', 'N/A')}")
        print(f"       Content safety: {result.get('ContentSafetyScorer', {}).get('content_safety', {}).get('mean', 'N/A')}")
        print(f"       Semantic similarity: {result.get('EmbeddingSimilarityScorer', {}).get('similarity_score', {}).get('mean', 'N/A')}")
        print(f"       LLM judge: {result.get('LLMJudgeScorer', {}).get('llm_judge', {}).get('mean', 'N/A')}")
    
    print(f"\n Step 2: Standard Evaluation Leaderboard Creation")
    
    # Create standard leaderboard with same four metrics (no latency as specified)
    if evaluation_objects:
        try:
            # Use ALL evaluations for comprehensive leaderboard comparison
            std_leaderboard = create_standard_leaderboard(evaluation_objects)
            
            if std_leaderboard:
                print(f"   Standard Leaderboard created: {std_leaderboard.uri()}")
                print(f"     Comparing: {len(evaluation_objects)} model variants")
                print(f"     Metrics: regulatory_compliance, content_safety, semantic_similarity, llm_judge") 
                print(f"     Note: Same four metrics as Part 2, no latency column")
            else:
                print(f"   Standard Leaderboard creation failed")
                
        except Exception as e:
            print(f"   Standard Leaderboard creation failed: {e}")
    
    print(f"\n Step 3: Evaluation Results Summary")
    
    # Create results summary
    print(f"   Model Performance Comparison:")
    for model_name, result in evaluation_results:
        # Extract mean scores for the four key metrics
        regulatory_score = result.get('PharmaceuticalQAScorer', {}).get('regulatory_compliance', {}).get('mean', 0)
        safety_score = result.get('ContentSafetyScorer', {}).get('content_safety', {}).get('mean', 0)
        similarity_score = result.get('EmbeddingSimilarityScorer', {}).get('similarity_score', {}).get('mean', 0)
        judge_score = result.get('LLMJudgeScorer', {}).get('llm_judge', {}).get('mean', 0)
        
        # Calculate overall score (same weighting as Part 2)
        overall_score = (regulatory_score * 0.4 + safety_score * 0.25 + 
                        similarity_score * 0.2 + judge_score * 0.15)
        
        print(f"     {model_name}: {overall_score:.3f}")
        print(f"       Regulatory: {regulatory_score:.3f}, Safety: {safety_score:.3f}")
        print(f"       Similarity: {similarity_score:.3f}, Judge: {judge_score:.3f}")
    
    print(f"\n Part 3 – Evaluations Complete!")
    print(f"   ✓ Systematic evaluation: {len(model_variants)} variants × {len(comprehensive_scorers)} scorers")
    print(f"   ✓ Same four metrics as Part 2: regulatory_compliance, content_safety, semantic_similarity, llm_judge")
    print(f"   ✓ No latency metric (removed as specified in playbook)")
    print(f"   ✓ Standard leaderboard mirrors Part 2 structure")
    print(f"   ✓ Enables apples-to-apples comparison with EvaluationLogger approach")
    
    return evaluation_results, evaluation_objects 