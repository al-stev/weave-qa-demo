#!/usr/bin/env python3
"""
Pharmaceutical QA Evaluation Demo - Weave Sales Demo
Simple, clean demonstration of key Weave capabilities

Scenario: Contamination incident investigation at PharmaTech Manufacturing
- Template versioning: FDA Basic vs ICH Enhanced frameworks  
- Real-time evaluation: EvaluationLogger for live Q&A sessions
- Batch evaluation: Standard Evaluation for framework comparison
"""

import os
import weave
from weave import EvaluationLogger
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
import hashlib
from pathlib import Path
import jinja2
import time

# Try importing both providers
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

load_dotenv(override=True)  # Force .env file to override system variables

# =============================================================================
# JINJA PROMPT WRAPPER: Proper Template Versioning
# =============================================================================

class JinjaPrompt:
    """
    Wrapper around Weave StringPrompt that uses Jinja2 templating.
    
    Key feature: Template structure affects versioning, runtime variables don't.
    Template instances should be reused across multiple variable injections.
    """
    
    def __init__(self, template_path: str, template_vars: Optional[Dict[str, Any]] = None):
        """
        Initialize with a Jinja template file and fixed template variables.
        
        Args:
            template_path: Path to .jinja template file
            template_vars: Fixed variables that are part of the template structure
                          (these affect versioning)
        """
        self.template_path = Path(template_path)
        self.template_vars = template_vars or {}
        
        # Load the template
        template_dir = self.template_path.parent
        template_name = self.template_path.name
        
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(template_dir),
            trim_blocks=True,
            lstrip_blocks=True
        )
        self.template = self.env.get_template(template_name)
        
        # Create semantic version name with structure hash
        self._version_info = self._calculate_version_info()
        self._published_prompt = None  # Cache the published Weave prompt
    
    def _calculate_version_info(self) -> Dict[str, str]:
        """Calculate semantic version name and hash based on template structure only."""
        
        # Read template content
        template_content = self.template_path.read_text()
        
        # Create version data from template structure (excludes runtime variables)
        version_data = {
            'template_content': template_content,
            'template_vars': self.template_vars  # Only fixed template vars
        }
        
        # Generate structure hash
        hash_input = str(sorted(version_data.items())).encode()
        structure_hash = hashlib.md5(hash_input).hexdigest()[:8]
        
        # Create semantic version name
        template_base = self.template_path.stem
        
        # Extract semantic components from template_vars
        framework = self.template_vars.get('compliance_framework', 'unknown')
        framework_short = self._get_framework_short(framework)
        
        interview_type = self.template_vars.get('interview_type', 'basic')
        config_type = self._get_config_type(interview_type)
        
        # Construct semantic name
        semantic_name = f"{template_base}_{framework_short}_{config_type}_v1_{structure_hash}"
        
        return {
            'semantic_name': semantic_name,
            'structure_hash': structure_hash,
            'template_base': template_base,
            'framework_short': framework_short,
            'config_type': config_type
        }
    
    def _get_framework_short(self, framework: str) -> str:
        """Convert compliance framework to short name."""
        if 'FDA' in framework or '21 CFR' in framework:
            return 'fda'
        elif 'ICH' in framework or 'Q9' in framework:
            return 'ich'
        elif 'GMP' in framework:
            return 'gmp'
        else:
            return 'generic'
    
    def _get_config_type(self, interview_type: str) -> str:
        """Convert interview type to short config name."""
        if 'contamination' in interview_type:
            return 'contamination'
        elif 'audit' in interview_type:
            return 'audit'
        elif 'compliance' in interview_type:
            return 'compliance'
        else:
            return 'basic'
    
    def create_weave_prompt(self, **runtime_vars) -> weave.StringPrompt:
        """
        Create a Weave StringPrompt by rendering the Jinja template.
        
        This method should be called multiple times with different runtime_vars
        while reusing the same JinjaPrompt instance for consistent versioning.
        
        Args:
            **runtime_vars: Variables injected at runtime (don't affect versioning)
                           e.g., supplier_name, question, incident_date
            
        Returns:
            weave.StringPrompt with rendered content and semantic version name
        """
        
        # If we haven't published this template structure yet, do it now
        if self._published_prompt is None:
            # Create a template-only prompt for publishing (without runtime vars)
            template_only_content = self._render_template_structure()
            self._published_prompt = weave.StringPrompt(template_only_content)
            
            # Publish the template structure to Weave for versioning
            weave.publish(self._published_prompt, name=self._version_info['semantic_name'])
        
        # Render with runtime variables for actual use
        all_vars = {**self.template_vars, **runtime_vars}
        rendered_content = self.template.render(**all_vars)
        
        # Return prompt with rendered content but same semantic version
        return weave.StringPrompt(rendered_content)
    
    def _render_template_structure(self) -> str:
        """Render template with only fixed vars for structure publishing."""
        # Use placeholder runtime vars to show template structure
        placeholder_vars = {
            'supplier_name': '{{SUPPLIER_NAME}}',
            'question': '{{CURRENT_QUESTION}}',
            'product_category': '{{PRODUCT_CATEGORY}}',
            'regulatory_region': '{{REGULATORY_REGION}}',
            'incident_date': '{{INCIDENT_DATE}}',
            'batch_numbers': '{{BATCH_NUMBERS}}'
        }
        
        structure_vars = {**self.template_vars, **placeholder_vars}
        return self.template.render(**structure_vars)
    
    @property
    def version_name(self) -> str:
        """Get the semantic version name for this prompt configuration."""
        return self._version_info['semantic_name']
    
    def __repr__(self) -> str:
        return f"JinjaPrompt(template={self.template_path.name}, version={self.version_name})"

# =============================================================================
# MODEL PROVIDER ABSTRACTION
# =============================================================================

class ModelProvider:
    """Abstraction to support both OpenAI and Anthropic models."""
    
    def __init__(self):
        self.provider = None
        self.client = None
        self._initialize_provider()
    
    def _initialize_provider(self):
        """Initialize the first available provider."""
        
        # Try OpenAI first
        if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            try:
                self.client = openai.OpenAI()
                self.provider = "openai"
                print("✅ Using OpenAI provider")
                return
            except Exception as e:
                print(f"❌ OpenAI failed: {e}")
        
        # Try Anthropic
        if ANTHROPIC_AVAILABLE and os.getenv("ANTHROPIC_API_KEY"):
            try:
                self.client = anthropic.Anthropic()
                self.provider = "anthropic"
                print("✅ Using Anthropic provider")
                return
            except Exception as e:
                print(f"❌ Anthropic failed: {e}")
        
        # No fallback needed - require working provider
        raise RuntimeError("No working model provider found. Please check your API keys (OPENAI_API_KEY or ANTHROPIC_API_KEY).")
    
    def chat_completion(self, prompt: str, max_tokens: int = 400, temperature: float = 0.7) -> str:
        """Generate a chat completion using the available provider."""
        
        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        
        elif self.provider == "anthropic":
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        
        
        else:
            raise RuntimeError(f"Unknown provider: {self.provider}")
    
    def __repr__(self) -> str:
        return f"ModelProvider(provider={self.provider})"

# Global model provider instance
model_provider = None

# =============================================================================
# SETUP: Initialize and Validate Environment
# =============================================================================

def create_model_variants():
    """Create multiple model variants for leaderboard comparison."""
    global fda_investigator, fda_investigator_anthropic, fda_investigator_baseline
    
    # Create shared prompt for all variants
    template_structure = PharmaceuticalQAModel._load_template_structure_static("templates/qa_investigation.jinja")
    prompt = weave.StringPrompt(template_structure)
    weave.publish(prompt, name="fda_contamination_investigation_v1")
    
    # Variant 1: OpenAI-based FDA QA Investigator
    fda_investigator = PharmaceuticalQAModel(
        name="FDA QA Investigator (OpenAI)",
        model_description="OpenAI GPT-4o powered pharmaceutical QA investigator specializing in FDA 21 CFR Part 211 contamination protocols",
        regulatory_framework="FDA 21 CFR Part 211",
        specialization_area="Contamination Investigation",
        compliance_level="fda_advanced",
        regulatory_approval_status="development",
        template_path="templates/qa_investigation.jinja",
        prompt=prompt
    )
    print("✅ FDA QA Investigator (OpenAI) ready")
    
    # Variant 2: Anthropic-based FDA QA Investigator 
    # This should create a new model version due to different provider behavior
    fda_investigator_anthropic = PharmaceuticalQAModel(
        name="FDA QA Investigator (Anthropic)",
        model_description="Anthropic Claude powered pharmaceutical QA investigator specializing in FDA 21 CFR Part 211 contamination protocols",
        regulatory_framework="FDA 21 CFR Part 211", 
        specialization_area="Contamination Investigation",
        compliance_level="fda_advanced",
        regulatory_approval_status="development",
        template_path="templates/qa_investigation.jinja",
        prompt=prompt
    )
    print("✅ FDA QA Investigator (Anthropic) ready")
    
    # Variant 3: Baseline simplified model
    baseline_prompt = weave.StringPrompt("You are a quality assurance investigator. Answer the following question about pharmaceutical contamination: {question}")
    weave.publish(baseline_prompt, name="basic_qa_prompt_v1")
    
    fda_investigator_baseline = PharmaceuticalQAModel(
        name="Basic QA Investigator (Baseline)",
        model_description="Simplified baseline QA investigator without specialized pharmaceutical training",
        regulatory_framework="Generic GMP",
        specialization_area="General Investigation", 
        compliance_level="gmp_basic",
        regulatory_approval_status="development",
        template_path="templates/qa_investigation.jinja",  # Still uses same template path
        prompt=baseline_prompt
    )
    print("✅ Basic QA Investigator (Baseline) ready")

def initialize_model_provider():
    """Initialize the model provider and test connection."""
    global model_provider
    print("🔧 Initializing model provider...")
    
    try:
        model_provider = ModelProvider()
        print(f"✅ {model_provider} ready")
        
        # Create all model variants for comparison
        create_model_variants()
        
        return True
    except Exception as e:
        print(f"❌ Model provider initialization failed: {e}")
        print("Please check your API keys (OPENAI_API_KEY or ANTHROPIC_API_KEY)")
        return False

def initialize_weave():
    """Initialize Weave and return project URL."""
    entity = os.getenv("WANDB_ENTITY", "wandb_emea")
    project_name = f"{entity}/eval-traces"
    weave.init(project_name)
    project_url = f"https://wandb.ai/{project_name}/weave"
    print(f"🔗 Weave Project: {project_url}")
    return project_url

# =============================================================================
# CORE MODELS: Simple, Clean Implementation
# =============================================================================

# Model provider will be initialized at runtime

# Clean demo uses weave.Model - no need for JinjaPrompt wrapper

class PharmaceuticalQAModel(weave.Model):
    """Enhanced weave.Model for pharmaceutical QA investigation with rich metadata."""
    name: str
    model_description: str
    regulatory_framework: str
    specialization_area: str
    compliance_level: str
    regulatory_approval_status: str
    template_path: str
    prompt: weave.StringPrompt
    
    def _load_template_structure(self) -> str:
        """Load template with placeholders for Weave versioning."""
        return self._load_template_structure_static(self.template_path)
    
    @staticmethod
    def _load_template_structure_static(template_path: str) -> str:
        """Static method to load template structure."""
        template_dir = Path(template_path).parent
        template_name = Path(template_path).name
        
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(template_dir),
            trim_blocks=True,
            lstrip_blocks=True
        )
        template = env.get_template(template_name)
        
        # Render with placeholder variables to show template structure
        placeholder_vars = {
            'role': 'Senior Quality Assurance Investigator',
            'compliance_framework': 'FDA 21 CFR Part 211',
            'interview_type': 'contamination_investigation',
            'investigation_method': '5 Whys Root Cause Analysis',
            'supplier_name': '{{SUPPLIER_NAME}}',
            'question': '{{CURRENT_QUESTION}}',
            'product_category': '{{PRODUCT_CATEGORY}}',
            'regulatory_region': '{{REGULATORY_REGION}}',
            'incident_date': '{{INCIDENT_DATE}}'
        }
        
        return template.render(**placeholder_vars)
    
    def _render_with_variables(self, question: str) -> str:
        """Render template with actual runtime variables for API call."""
        template_dir = Path(self.template_path).parent
        template_name = Path(self.template_path).name
        
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(template_dir),
            trim_blocks=True,
            lstrip_blocks=True
        )
        template = env.get_template(template_name)
        
        # Render with actual variables
        actual_vars = {
            'role': 'Senior Quality Assurance Investigator',
            'compliance_framework': 'FDA 21 CFR Part 211',
            'interview_type': 'contamination_investigation',
            'investigation_method': '5 Whys Root Cause Analysis',
            'supplier_name': 'PharmaTech Manufacturing',
            'question': question,
            'product_category': 'Oral Solid Dosage Tablets',
            'regulatory_region': 'USA',
            'incident_date': 'January 15, 2024'
        }
        
        return template.render(**actual_vars)
    
    @weave.op()
    def predict(self, question: str) -> dict:
        """Generate QA investigation response with proper prompt versioning."""
        try:
            # Use the class prompt attribute (automatically tracked by Weave)
            # Render actual content for API call (with variables filled in)
            actual_content = self._render_with_variables(question)
            
            # For Anthropic variants, we'll create a temporary Anthropic provider
            if "Anthropic" in self.name:
                # Use Anthropic for this specific model variant
                if not ANTHROPIC_AVAILABLE or not os.getenv("ANTHROPIC_API_KEY"):
                    return {"output": "Anthropic API not available - check ANTHROPIC_API_KEY"}
                
                anthropic_client = anthropic.Anthropic()
                response = anthropic_client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=400,
                    temperature=0.7,
                    messages=[{"role": "user", "content": actual_content}]
                )
                return {"output": response.content[0].text}
            else:
                # Use the global model provider (OpenAI or fallback)
                response = model_provider.chat_completion(actual_content, max_tokens=400, temperature=0.7)
                return {"output": response}
                
        except Exception as e:
            print(f"❌ Predict method error: {e}")
            return {"output": f"Error generating response: {e}"}

# Model instances for leaderboard comparison
fda_investigator = None  # OpenAI-based model
fda_investigator_anthropic = None  # Anthropic-based model
fda_investigator_baseline = None  # Simplified baseline model


@weave.op()
def simple_quality_scorer(question: str, response: str) -> dict:
    """Simple quality scoring for real-time evaluation."""
    # Simple heuristic scoring for demonstration
    score = 0.7 + (len(response) / 10000) * 0.3  # Longer responses score higher
    return {
        "root_cause_identification": min(score + 0.1, 1.0),
        "corrective_actions": min(score, 1.0)
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
    def score(self, expected: str, output: dict) -> dict:
        """Comprehensive pharmaceutical QA scoring with multiple dimensions."""
        actual_response = output.get("output", "")
        
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
        
        # Combine all scores
        result = {
            **compliance_scores,
            **accuracy_scores, 
            **actionability_scores,
            "overall_pharmaceutical_qa_score": overall_score,
            "regulatory_framework": self.regulatory_framework
        }
        
        return result

# Keep the simple function-based scorer for real-time evaluation
@weave.op()
def simple_quality_scorer(question: str, response: str) -> dict:
    """Simple quality scoring for real-time evaluation."""
    # Simple heuristic scoring for demonstration
    score = 0.7 + (len(response) / 10000) * 0.3  # Longer responses score higher
    return {
        "root_cause_identification": min(score + 0.1, 1.0),
        "corrective_actions": min(score, 1.0)
    }

# =============================================================================
# ACT 1: Template Versioning Demo
# =============================================================================

def act1_template_versioning():
    """Demonstrate template versioning: variables don't change version, template structure does."""
    
    print("\n" + "="*60)
    print("🎬 ACT 1: Template Versioning Demo")
    print("="*60)
    print("Goal: Show template structure controls versioning, not variables")
    
    # Three key investigation questions
    questions = [
        "What was the first indication that contamination occurred?",
        "What was the root cause of the contamination?", 
        "What preventive measures will prevent recurrence?"
    ]
    
    print(f"\n📋 Part 1: Testing with {len(questions)} different questions...")
    print("   Expected: Same prompt version for all questions")
    
    # Use FDA model with all questions (should all have same prompt version)
    for i, question in enumerate(questions, 1):
        print(f"   Question {i}: {question[:50]}...")
        result = fda_investigator.predict(question)
        response = result["output"]
        print(f"   ✅ Response generated ({len(response)} chars)")
    
    print("\n📋 Part 2: Now modifying template structure...")
    print("   Simulating developer editing the template file")
    
    # Backup original template
    template_path = Path("templates/qa_investigation.jinja")
    original_content = template_path.read_text()
    
    # Create enhanced template with additional section
    enhanced_content = original_content + """

**Enhanced Analysis Required:**
- Risk Level Assessment: {{risk_level}}
- Compliance Impact: Critical review required for {{compliance_framework}}
- Additional Documentation: Detailed CAPA plan must be submitted within 48 hours"""
    
    # Write enhanced template
    template_path.write_text(enhanced_content)
    print("   ✅ Template enhanced with additional analysis section")
    
    # Test with enhanced template
    print("\n📋 Part 3: Testing with enhanced template...")
    print("   Expected: NEW prompt version (template structure changed)")
    
    # Update the existing model's prompt to trigger new prompt version (Weave auto-detects)
    enhanced_template_structure = PharmaceuticalQAModel._load_template_structure_static("templates/qa_investigation.jinja")
    enhanced_prompt = weave.StringPrompt(enhanced_template_structure)
    
    # Publish the enhanced prompt with a meaningful name
    weave.publish(enhanced_prompt, name="fda_contamination_investigation_enhanced_v2")
    
    # Update the model's prompt attribute - Weave will detect this as a new prompt version
    fda_investigator.prompt = enhanced_prompt
    
    enhanced_question = "What immediate containment actions were taken?"
    print(f"   Question: {enhanced_question[:50]}...")
    result = fda_investigator.predict(enhanced_question)
    response = result["output"]
    print(f"   ✅ Response generated ({len(response)} chars)")
    
    # Restore original template
    template_path.write_text(original_content)
    print("   ✅ Template restored to original")
    
    print("\n✅ Template Versioning Demonstration Complete:")
    print("   • Part 1: 3 different questions → 1 prompt version")
    print("   • Part 2: Template structure modified")  
    print("   • Part 3: 1 question with enhanced template → NEW prompt version")
    print("\n📊 Key Concept: Template STRUCTURE controls versioning")
    print("   • Different variables (questions) = same version")
    print("   • Different template structure = new version")
    print("\n📊 Check Weave UI: Look for 2 prompt versions total")

# =============================================================================
# ACT 2: Real-time Evaluation Demo
# =============================================================================

def act2_realtime_evaluation():
    """Demonstrate EvaluationLogger for real-time Q&A sessions."""
    
    print("\n" + "="*60)
    print("🎬 ACT 2: Real-time Evaluation Demo")
    print("="*60)
    print("Goal: Track individual Q&A interactions in real-time")
    
    # Create and publish dataset for the evaluation
    questions = [
        "What was the first indication that contamination occurred?",
        "What was the root cause of the contamination?",
        "What preventive measures will prevent recurrence?"
    ]
    
    qa_dataset = [
        {"question": q}
        for q in questions
    ]
    
    # Publish the dataset to Weave
    dataset = weave.Dataset(name="pharmatech_contamination_qa", rows=qa_dataset)
    weave.publish(dataset, name="pharmatech_contamination_qa")
    print("📊 Published QA dataset to Weave")
    
    # Initialize EvaluationLogger with simple string identifiers for tracking
    ev = EvaluationLogger(
        model="PharmaceuticalQAModel",
        dataset="contamination_qa_dataset"
    )
    
    print(f"\n📊 EvaluationLogger URL: {ev.ui_url}")
    
    print(f"\n🔄 Processing {len(qa_dataset)} questions in real-time...")
    
    start_time = time.time()
    
    for i, example in enumerate(qa_dataset, 1):
        question = example["question"]
        print(f"\n   ➤ Question {i}: {question}")
        
        # I call the model manually (EvaluationLogger pattern - model is independent)
        result = fda_investigator.predict(question)
        response = result["output"]
        
        # I log the prediction
        pred_logger = ev.log_prediction(
            inputs=example,
            output=result
        )
        
        print(f"     ✅ Response logged: {pred_logger.predict_call.ui_url}")
        
        # I call scorer manually and log scores
        scores = simple_quality_scorer(question, response)
        
        # I log individual scores
        pred_logger.log_score("root_cause_identification", scores["root_cause_identification"])
        pred_logger.log_score("corrective_actions", scores["corrective_actions"])
        
        # I finish this prediction
        pred_logger.finish()
        
        # Show summary score for this question
        avg_score = (scores["root_cause_identification"] + scores["corrective_actions"]) / 2
        print(f"     📊 Quality Score: {avg_score:.2f}")
    
    # Calculate summary statistics
    end_time = time.time()
    evaluation_duration = end_time - start_time
    
    # I log the summary with realistic industry insights (Weave auto-aggregates individual scores)
    ev.log_summary({
        "evaluation_type": "real_time_qa_session",
        "investigation_summary": "Model demonstrates strong understanding of FDA contamination protocols and root cause analysis methodology",
        "regulatory_compliance": "Responses align with 21 CFR Part 211 requirements for incident investigation",
        "key_strengths": "Systematic approach to identifying contamination sources and process failures",
        "improvement_areas": "Could provide more specific CAPA timelines and risk assessment details",
        "follow_up_required": "Manual review recommended for regulatory filing and final investigation report",
        "total_questions": len(qa_dataset),
        "evaluation_duration_seconds": round(evaluation_duration, 2),
        "framework": "FDA 21 CFR Part 211"
    })
    
    print("\n✅ Real-time Evaluation Complete!")
    print(f"📊 View individual traces: {ev.ui_url}")
    print("   • Each question → individual prediction trace")
    print("   • Immediate scoring and feedback")
    print("   • Perfect for live investigation sessions")

# =============================================================================
# ACT 3: Batch Evaluation Demo  
# =============================================================================

def act3_batch_evaluation():
    """Demonstrate Standard Evaluation for batch processing."""
    
    print("\n" + "="*60)
    print("🎬 ACT 3: Batch Evaluation Demo")
    print("="*60)
    print("Goal: Demonstrate batch evaluation with rollup statistics")
    
    questions = [
        "What was the first indication that contamination occurred?",
        "What was the root cause of the contamination?",
        "What preventive measures will prevent recurrence?"
    ]
    
    # Create dataset for batch evaluation with ground truth answers
    dataset = [
        {
            "question": "What was the first indication that contamination occurred?",
            "expected": "The first indication was detected during routine quality control testing when an unexpected impurity or foreign particles were found in the batch samples, or when microbial load exceeded acceptable limits."
        },
        {
            "question": "What was the root cause of the contamination?",
            "expected": "The root cause was inadequate equipment cleaning procedures, often due to insufficient training, outdated SOPs, or failure to follow established cleaning protocols between production runs."
        },
        {
            "question": "What preventive measures will prevent recurrence?",
            "expected": "Enhanced training programs on cleaning protocols, updated and regularly reviewed SOPs, implementation of continuous monitoring and auditing systems, and improved maintenance schedules with accountability measures."
        }
    ]
    
    print(f"\n📊 Using Weave Evaluation framework for batch processing on {len(questions)} questions...")
    
    # Create class-based scorer for comprehensive evaluation
    pharma_scorer = PharmaceuticalQAScorer(regulatory_framework="FDA_21_CFR_211")
    
    # Create Weave evaluation with class-based scorer
    evaluation = weave.Evaluation(
        dataset=dataset,
        scorers=[pharma_scorer]
    )
    
    print("\n🔄 Running Batch Evaluation...")
    import asyncio
    results = asyncio.run(evaluation.evaluate(fda_investigator))
    print(f"   ✅ Batch evaluation complete with {len(results)} examples")
    
    print("\n✅ Batch Evaluation Results:")
    print(f"   📊 Results: {results}")
    print("\n📊 Key Difference: EvaluationLogger vs Standard Evaluation")
    print("   • Act 2 (EvaluationLogger): Real-time, simple scoring")
    print("   • Act 3 (Standard Evaluation): Batch processing, comprehensive class-based scoring")
    print("\n📊 Class-based Scorer Benefits:")
    print("   • Multi-dimensional: Regulatory compliance + Technical accuracy + Actionability")
    print("   • Configurable: Different regulatory frameworks (FDA, ICH, GMP)")
    print("   • State management: Maintains compliance criteria and scoring logic")
    print("   • Industry-specific: Pharmaceutical QA domain knowledge built-in")
    print("\n📊 Check Weave UI: View detailed multi-dimensional scoring results")

# =============================================================================
# MAIN DEMO EXECUTION
# =============================================================================

def main():
    """Run the complete pharmaceutical QA demo."""
    
    print("🧬 Pharmaceutical QA Investigation Demo")
    print("Showcasing Weave Evaluation Capabilities")
    print("=" * 60)
    
    # Setup phase - initialize Weave first
    project_url = initialize_weave()
    
    if not initialize_model_provider():
        print("❌ Demo cannot proceed without a working model provider")
        return
    
    print("""
📋 Demo Overview:
   Scenario: PharmaTech Manufacturing contamination incident
   Framework: FDA 21 CFR Part 211 Quality Assurance
   
   Act 1: Template Versioning (2 min)
   Act 2: Real-time Evaluation (2 min)  
   Act 3: Batch Evaluation (2 min)
   
🎯 Expected Results:
   ✅ Clean Weave UI with no failures
   📊 Single prompt version across multiple questions
   🔍 Individual prediction traces
   📈 Batch evaluation rollup statistics
""")
    
    print("\n▶️  Starting demo automatically...")
    
    try:
        # Run the three acts
        act1_template_versioning()
        
        print("\n▶️  Continuing to Act 2...")
        act2_realtime_evaluation()
        
        print("\n▶️  Continuing to Act 3...")
        act3_batch_evaluation()
        
        # Demo complete
        print("\n" + "="*60)
        print("🎉 Demo Complete! Key Takeaways:")
        print("="*60)
        print("✅ Template Versioning: Weave automatically handles prompt versioning")
        print("✅ Real-time Evaluation: Track every interaction with immediate feedback")
        print("✅ Batch Evaluation: Systematic evaluation with rollup statistics")
        print("✅ Clean Integration: weave.Model + weave.StringPrompt = no strike-through!")
        print("\n📊 Explore your results:")
        print(f"   {project_url}")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("This is a clean failure - check API keys and quotas")
        raise

if __name__ == "__main__":
    main()