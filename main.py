"""
QA Demo – Main CLI Entry Point

Demonstrates Weave's evaluation capabilities through pharmaceutical quality assurance scenarios.
Supports four modes: prompt versioning, evaluation logger, comprehensive evaluations, and all combined.

Usage:
    python main.py --mode prompt_versioning  # Part 1 only
    python main.py --mode eval_logger        # Part 1 + Part 2 (default)
    python main.py --mode evaluation         # Part 1 + Part 3
    python main.py --mode all                # Parts 1 + 2 + 3
    python main.py --project-name my-project # Custom project name
    python main.py                           # Default: eval_logger
"""

import os
import argparse
import weave
from dotenv import load_dotenv

from part1_prompt_versioning import part1_prompt_versioning
from part2_evaluation_logger import part2_evaluation_logger  
from part3_evaluation import part3_evaluation
from models import initialize_model_provider


def initialize_weave(project_name="weave-evals-traces"):
    """Initialize Weave and return project URL."""
    entity = os.getenv("WANDB_ENTITY", "wandb_emea")
    full_project_name = f"{entity}/{project_name}"
    weave.init(full_project_name)
    project_url = f"https://wandb.ai/{full_project_name}/weave"
    print(f"Weave project URL: {project_url}")
    return project_url


def setup_environment():
    """Set up environment and verify API keys."""
    load_dotenv(override=True)  # Force .env file to override system variables
    
    # Verify API key configuration
    openai_key = os.getenv("OPENAI_API_KEY", "NOT_SET")
    print(f"Using OpenAI API Key: {openai_key[:20]}... (from .env)")
    if openai_key.startswith("sk-proj-"):
        print("Confirmed: Using project API key from .env file")
    elif openai_key.startswith("sk-svcacct-"):
        print("Warning: Using service account key - may have rate limits")
    else:
        print("Warning: Unexpected API key format")


def main():
    """Main CLI entry point with mode selection."""
    parser = argparse.ArgumentParser(
        description="QA Demo - Weave evaluation capabilities showcase",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode prompt_versioning    # Part 1: Prompt versioning demo
  python main.py --mode eval_logger          # Part 1+2: EvaluationLogger workflow  
  python main.py --mode evaluation           # Part 1+3: Comprehensive evaluation
  python main.py --mode all                  # Parts 1+2+3: Complete showcase
  python main.py --project-name my-demo      # Custom project name
  python main.py                             # Default: eval_logger
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["prompt_versioning", "eval_logger", "evaluation", "all"],
        default="eval_logger",
        help="Demo mode to run (default: eval_logger)"
    )
    
    parser.add_argument(
        "--project-name",
        default="weave-evals-traces",
        help="Weave project name (default: weave-evals-traces)"
    )
    
    args = parser.parse_args()
    
    print("QA Demo – Weave Evaluation Capabilities")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Project: {args.project_name}")
    
    # Setup environment and initialize Weave
    setup_environment()
    project_url = initialize_weave(args.project_name)
    
    # Initialize model provider
    if not initialize_model_provider():
        print("Demo cannot proceed without a working model provider")
        print("Please check your API keys (OPENAI_API_KEY or ANTHROPIC_API_KEY)")
        return
    
    print("""
Demo Overview:
  Scenario: PharmaTech Manufacturing contamination incident
  Framework: FDA 21 CFR Part 211 Quality Assurance
  
  Part 1: Prompt Versioning Showcase
  Part 2: Evaluation Logger ("a more flexible approach to evaluating AI applications")  
  Part 3: Evaluations ("rigorous evaluations of AI applications")
  
Expected Results:
  • Clean Weave UI with no failures or strikethrough operations
  • Automatic prompt versioning demonstration
  • Multi-model evaluation with four metrics (no latency)
  • Comprehensive leaderboard comparisons
""")
    
    try:
        if args.mode == "prompt_versioning":
            # Run Part 1 only
            print(f"\nRunning Part 1 – Prompt Versioning Showcase")
            part1_prompt_versioning()
            
        elif args.mode == "eval_logger":
            # Run Part 1 + Part 2 (default mode)
            print(f"\nRunning Part 1 + Part 2 – Prompt Versioning + Evaluation Logger")
            part1_prompt_versioning()
            part2_evaluation_logger()
            
        elif args.mode == "evaluation":
            # Run Part 1 + Part 3
            print(f"\nRunning Part 1 + Part 3 – Prompt Versioning + Comprehensive Evaluations")
            part1_prompt_versioning()
            evaluation_results, evaluation_objects = part3_evaluation()
            
        elif args.mode == "all":
            # Run all three parts
            print(f"\nRunning All Parts – Complete Weave Evaluation Showcase")
            print("\n" + "="*60)
            print("PART 1 – PROMPT VERSIONING")
            print("="*60)
            part1_prompt_versioning()
            
            print("\n" + "="*60)
            print("PART 2 – EVALUATION LOGGER")
            print("="*60)
            part2_evaluation_logger()
            
            print("\n" + "="*60)
            print("PART 3 – COMPREHENSIVE EVALUATIONS")
            print("="*60)
            evaluation_results, evaluation_objects = part3_evaluation()
        
        # Demo complete
        print("\n" + "="*60)
        print("QA Demo Complete! Key Takeaways:")
        print("="*60)
        
        if args.mode == "prompt_versioning":
            print(" ✓ Prompt Versioning: Weave StringPrompt automatically tracks structural changes")
        elif args.mode == "eval_logger":
            print(" ✓ Prompt Versioning: Weave StringPrompt automatically tracks structural changes")
            print(" ✓ Evaluation Logger: Multi-model evaluation with real-time scoring")
            print(" ✓ Four Metrics: regulatory_compliance, content_safety, semantic_similarity, llm_judge")
            print(" ✓ EL Leaderboard: Comprehensive comparison without latency metric")
        elif args.mode == "evaluation":
            print(" ✓ Prompt Versioning: Weave StringPrompt automatically tracks structural changes")
            print(" ✓ Comprehensive Evaluation: Systematic multi-model comparison")
            print(" ✓ Four Metrics: Same as Part 2 for apples-to-apples comparison")
            print(" ✓ Standard Leaderboard: Rigorous evaluation workflow")
        elif args.mode == "all":
            print(" ✓ Prompt Versioning: Weave StringPrompt automatically tracks structural changes")
            print(" ✓ Evaluation Logger: Multi-model evaluation with real-time scoring")
            print(" ✓ Comprehensive Evaluation: Systematic multi-model comparison")
            print(" ✓ Four Metrics: Complete scoring across all evaluation methods")
            print(" ✓ Complete Comparison: Both EL and Standard leaderboards")
            print(" ✓ Full Workflow: End-to-end Weave evaluation capabilities")
        
        print(" ✓ Clean Integration: All operations functional with proper instrumentation")
        print(" ✓ No Latency Metrics: Removed as specified in requirements")
        print(f"\nExplore your results: {project_url}")
        
    except Exception as e:
        print(f"\nDemo failed: {e}")
        print("This indicates an issue with API keys, quotas, or configuration")
        raise


if __name__ == "__main__":
    main() 