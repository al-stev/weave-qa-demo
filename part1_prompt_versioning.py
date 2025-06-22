"""
Part 1 – Prompt Versioning Showcase

Learning Objective:
Demonstrate that a single StringPrompt name accrues new versions only when its 
template structure changes, not when runtime variables change.

This part showcases Weave's automatic prompt versioning by creating two versions
of the same prompt with different methodologies.
"""

import weave
from pathlib import Path
import jinja2
from models import PharmaceuticalQAModel
from scorers import simple_quality_scorer


def part1_prompt_versioning():
    """Demonstrate template versioning: variables don't change version, template structure does."""
    
    print("\n" + "="*60)
    print(" Part 1 – Prompt Versioning Showcase")
    print("="*60)
    print("Goal: Show automatic versioning with prompt improvements")
    
    print("\n Step 1: Create Version 1 (Weaker Prompt - No 5 Whys)...")
    print("   Creating baseline prompt without structured methodology")
    
    # Create weaker prompt by removing 5 Whys methodology
    def _load_template_structure_weaker(template_path: str) -> str:
        """Load template structure without 5 Whys methodology."""
        template_dir = Path(template_path).parent
        template_name = Path(template_path).name
        
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(template_dir),
            trim_blocks=True,
            lstrip_blocks=True
        )
        template = env.get_template(template_name)
        
        # Render with placeholder variables - NO 5 Whys methodology
        placeholder_vars = {
            'role': 'Senior Quality Assurance Investigator',
            'compliance_framework': 'FDA 21 CFR Part 211',
            'interview_type': 'contamination_investigation',
            'investigation_method': 'General Investigation',  # Generic instead of 5 Whys
            'supplier_name': '{{SUPPLIER_NAME}}',
            'question': '{{CURRENT_QUESTION}}',
            'product_category': '{{PRODUCT_CATEGORY}}',
            'regulatory_region': '{{REGULATORY_REGION}}',
            'incident_date': '{{INCIDENT_DATE}}'
        }
        
        return template.render(**placeholder_vars)
    
    # Create and publish weaker prompt version
    weaker_template_structure = _load_template_structure_weaker("qa_demo/templates/qa_investigation.jinja")
    weaker_prompt = weave.StringPrompt(weaker_template_structure)
    weave.publish(weaker_prompt, name="qa_contamination_investigation")
    
    # Create temporary model with weaker prompt
    weaker_model = PharmaceuticalQAModel(
        name="QA Investigator (Version 1 - Generic)",
        model_description="QA investigator with generic investigation methodology",
        regulatory_framework="FDA 21 CFR Part 211",
        specialization_area="Contamination Investigation",
        compliance_level="fda_basic",
        regulatory_approval_status="development",
        template_path="qa_demo/templates/qa_investigation.jinja",
        prompt=weaker_prompt
    )
    
    # Test with weaker prompt
    test_question = "What was the root cause of the contamination?"
    print(f"   Question: {test_question}")
    response_v1 = weaker_model.predict(input=test_question)
    print(f"    Version 1 Response generated ({len(response_v1)} chars)")
    
    # Score the weaker version
    scores_v1 = simple_quality_scorer(test_question, response_v1)
    print(f"    Version 1 Quality Score: {(scores_v1['root_cause_identification'] + scores_v1['corrective_actions']) / 2:.3f}")
    
    print("\n Step 2: Create Version 2 (Enhanced Prompt - With 5 Whys)...")
    print("   Creating improved prompt with structured methodology")
    
    # Create enhanced prompt with 5 Whys methodology (using original template structure)
    enhanced_template_structure = PharmaceuticalQAModel._load_template_structure_static("qa_demo/templates/qa_investigation.jinja")
    enhanced_prompt = weave.StringPrompt(enhanced_template_structure)
    
    # Publish under SAME NAME - Weave will create Version 2 automatically
    weave.publish(enhanced_prompt, name="qa_contamination_investigation")
    print("    Enhanced prompt published with 5 Whys methodology")
    
    # Create model with enhanced prompt (this uses the original template with 5 Whys)
    enhanced_model = PharmaceuticalQAModel(
        name="QA Investigator (Version 2 - Structured)",
        model_description="QA investigator with 5 Whys root cause analysis methodology",
        regulatory_framework="FDA 21 CFR Part 211",
        specialization_area="Contamination Investigation",
        compliance_level="fda_advanced",
        regulatory_approval_status="development",
        template_path="qa_demo/templates/qa_investigation.jinja",
        prompt=enhanced_prompt
    )
    
    # Test with enhanced prompt (same question for comparison)
    print(f"   Question: {test_question}")
    response_v2 = enhanced_model.predict(input=test_question)
    print(f"    Version 2 Response generated ({len(response_v2)} chars)")
    
    # Score the enhanced version
    scores_v2 = simple_quality_scorer(test_question, response_v2)
    print(f"    Version 2 Quality Score: {(scores_v2['root_cause_identification'] + scores_v2['corrective_actions']) / 2:.3f}")
    
    print("\n Step 3: Version Comparison Results...")
    improvement = ((scores_v2['root_cause_identification'] + scores_v2['corrective_actions']) / 2) - ((scores_v1['root_cause_identification'] + scores_v1['corrective_actions']) / 2)
    print(f"    Quality Improvement: {improvement:.3f} ({improvement*100:+.1f}%)")
    
    if improvement > 0:
        print("    Version 2 (5 Whys) outperforms Version 1 (Generic)")
    else:
        print("     Unexpected: Version 1 scored higher than Version 2")
    
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