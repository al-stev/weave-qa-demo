"""
Part 1 – Prompt Versioning Showcase

Demonstrates that Weave automatically versions prompts when template structure changes, 
not when runtime variables change.
"""

import weave
from pathlib import Path
import jinja2
from models import PharmaceuticalQAModel
from scorers import simple_quality_scorer

# Constants
TEST_QUESTION = "What was the root cause of the contamination?"

# Placeholder variables for template structure (not runtime values)
PLACEHOLDER_VARS = {
    'role': 'Senior Quality Assurance Investigator',
    'compliance_framework': 'FDA 21 CFR Part 211',
    'interview_type': 'contamination_investigation',
    'supplier_name': '{{SUPPLIER_NAME}}',
    'question': '{{CURRENT_QUESTION}}',
    'product_category': '{{PRODUCT_CATEGORY}}',
    'regulatory_region': '{{REGULATORY_REGION}}',
    'incident_date': '{{INCIDENT_DATE}}'
}

# Display labels
LABELS = {
    'baseline': 'baseline (generic methodology)',
    'enhanced': 'enhanced (5 Whys methodology)'
}


def load_baseline_template(template_path: str) -> str:
    """Load template structure without 5 Whys methodology for baseline version."""
    template_dir = Path(template_path).parent
    template_name = Path(template_path).name
    
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(template_dir),
        trim_blocks=True,
        lstrip_blocks=True
    )
    template = env.get_template(template_name)
    
    # Create baseline placeholder vars with generic investigation method
    baseline_vars = PLACEHOLDER_VARS.copy()
    baseline_vars['investigation_method'] = 'General Investigation'  # Generic instead of 5 Whys
    
    return template.render(**baseline_vars)


def _avg_root_and_corrective(scores: dict) -> float:
    """Calculate average of root cause identification and corrective actions scores."""
    return (scores['root_cause_identification'] + scores['corrective_actions']) / 2


def part1_prompt_versioning():
    """Demonstrate template versioning: variables don't change version, template structure does."""
    
    try:
        print("\n" + "="*60)
        print(" Part 1 – Prompt Versioning Showcase")
        print("="*60)
        print("Goal: Show automatic versioning with prompt improvements")
        
        print(f"\n Step 1: Create Version 1 ({LABELS['baseline']})...")
        
        # Create baseline prompt by removing 5 Whys methodology
        baseline_template_structure = load_baseline_template("templates/qa_investigation.jinja")
        baseline_prompt = weave.StringPrompt(baseline_template_structure)
        baseline_version_id = weave.publish(baseline_prompt, name="qa_contamination_investigation")
        print(f"   Baseline prompt published (Version ID: {baseline_version_id})")
        
        # Create temporary model with baseline prompt
        baseline_model = PharmaceuticalQAModel(
            name="QA Investigator (Version 1 - Generic)",
            model_description="QA investigator with generic investigation methodology",
            regulatory_framework="FDA 21 CFR Part 211",
            specialization_area="Contamination Investigation",
            compliance_level="fda_basic",
            regulatory_approval_status="development",
            template_path="templates/qa_investigation.jinja",
            prompt=baseline_prompt
        )
        
        # Test with baseline prompt
        print(f"   Question: {TEST_QUESTION}")
        response_v1 = baseline_model.predict(input=TEST_QUESTION)
        print(f"   Version 1 Response generated ({len(response_v1)} chars)")
        
        # Score the baseline version
        scores_v1 = simple_quality_scorer(TEST_QUESTION, response_v1)
        avg_score_v1 = _avg_root_and_corrective(scores_v1)
        print(f"   Version 1 Quality Score: {avg_score_v1:.3f}")
        
        print(f"\n Step 2: Create Version 2 ({LABELS['enhanced']})...")
        
        # Create enhanced prompt with 5 Whys methodology (using original template structure)
        enhanced_template_structure = PharmaceuticalQAModel._load_template_structure_static("templates/qa_investigation.jinja")
        enhanced_prompt = weave.StringPrompt(enhanced_template_structure)
        
        # Publish under SAME NAME - Weave will create Version 2 automatically
        # NOTE: Runtime variables don't create new prompt versions - only template structure changes do
        enhanced_version_id = weave.publish(enhanced_prompt, name="qa_contamination_investigation")
        print(f"   Enhanced prompt published (Version ID: {enhanced_version_id})")
        
        # Create model with enhanced prompt (this uses the original template with 5 Whys)
        enhanced_model = PharmaceuticalQAModel(
            name="QA Investigator (Version 2 - Structured)",
            model_description="QA investigator with 5 Whys root cause analysis methodology",
            regulatory_framework="FDA 21 CFR Part 211",
            specialization_area="Contamination Investigation",
            compliance_level="fda_advanced",
            regulatory_approval_status="development",
            template_path="templates/qa_investigation.jinja",
            prompt=enhanced_prompt
        )
        
        # Test with enhanced prompt (same question for comparison)
        print(f"   Question: {TEST_QUESTION}")
        response_v2 = enhanced_model.predict(input=TEST_QUESTION)
        print(f"   Version 2 Response generated ({len(response_v2)} chars)")
        
        # Score the enhanced version
        scores_v2 = simple_quality_scorer(TEST_QUESTION, response_v2)
        avg_score_v2 = _avg_root_and_corrective(scores_v2)
        print(f"   Version 2 Quality Score: {avg_score_v2:.3f}")
        
        print("\n Step 3: Version Comparison Results...")
        improvement = avg_score_v2 - avg_score_v1
        print(f"   Quality Improvement: {improvement:.3f} ({improvement*100:+.1f}%)")
        
        if improvement > 0:
            print("   Version 2 (5 Whys) outperforms Version 1 (Generic)")
        else:
            print("   Unexpected: Version 1 scored higher than Version 2")
        
        print("\n Prompt Versioning Demonstration Complete:")
        print("   • Step 1: Generic investigation methodology → Version 1")
        print("   • Step 2: 5 Whys structured methodology → Version 2")  
        print("   • Step 3: Measurable quality improvement demonstrated")
        print("\n Key Concept: Weave AUTOMATIC versioning")
        print("   • Same prompt name with different content = automatic versioning")
        print("   • Structured methodology improves regulatory compliance")
        print("\n Check Weave UI: 'qa_contamination_investigation' shows 2 versions")
        print("   • Version 1: Generic Investigation methodology")
        print("   • Version 2: 5 Whys Root Cause Analysis methodology")
        
    except FileNotFoundError as e:
        print(f"\n❌ Template file not found: {e}")
        print("   Ensure 'templates/qa_investigation.jinja' exists in the project directory")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("   Check your API keys and network connection")
        raise 