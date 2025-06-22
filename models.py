"""
Model classes for QA Demo.

Contains the ModelProvider abstraction and PharmaceuticalQAModel implementation.
"""

import os
import weave
from pathlib import Path
import jinja2

# =============================================================================
# MODEL PROVIDER ABSTRACTION (with proper @weave.op decoration)  
# =============================================================================

class ModelProvider:
    """Abstraction to support both OpenAI and Anthropic models."""
    
    def __init__(self):
        self.provider = None
        self.client = None
        self._initialize_provider()
    
    def _initialize_provider(self):
        """Initialize the first available provider."""
        
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
        
        # Try OpenAI first
        if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            try:
                self.client = openai.OpenAI()
                self.provider = "openai"
                print(" Using OpenAI provider")
                return
            except Exception as e:
                print(f" OpenAI failed: {e}")
        
        # Try Anthropic
        if ANTHROPIC_AVAILABLE and os.getenv("ANTHROPIC_API_KEY"):
            try:
                self.client = anthropic.Anthropic()
                self.provider = "anthropic"
                print(" Using Anthropic provider")
                return
            except Exception as e:
                print(f" Anthropic failed: {e}")
        
        # Require at least one working provider for demo
        raise RuntimeError("No working model provider found. Please check your API keys (OPENAI_API_KEY or ANTHROPIC_API_KEY).")
    
    @weave.op()  # Decorated for proper Weave call graph tracking
    def chat_completion(self, prompt: str, max_tokens: int = 400, temperature: float = 0.7) -> str:
        """Generate a chat completion using the available provider."""
        
        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            content = response.choices[0].message.content
            if content is None:
                raise RuntimeError("OpenAI API returned None content - check API quota/rate limits")
            return content
        
        elif self.provider == "anthropic":
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            content = response.content[0].text
            if content is None:
                raise RuntimeError("Anthropic API returned None content - check API quota/rate limits")
            return content
        
        else:
            raise RuntimeError(f"Unknown provider: {self.provider}")
    
    def __repr__(self) -> str:
        return f"ModelProvider(provider={self.provider})"


# =============================================================================
# PHARMACEUTICAL QA MODELS: Enhanced weave.Model implementation with rich metadata
# =============================================================================

class PharmaceuticalQAModel(weave.Model):
    """Enhanced weave.Model representing a pharma-QA investigator.

    Why use `weave.Model` here?
    • The prompt and every call of `predict` are auto-logged as **traceable
      objects**.
    • When we instantiate the model, Weave snapshots the prompt version so
      later edits don't retro-actively change historical evaluations.
    • Metadata (framework, compliance_level, etc.) is stored alongside the
      model, making leaderboards filterable by these fields in the UI.
    """
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
    
    # --- Prompt Versioning ---
    # We freeze the *template structure* here. Publishing another
    # StringPrompt with the same name but different structure ⇒ Weave
    # stores a new prompt **version**. Runtime variable values below do
    # NOT create new versions.
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
    
    @weave.op()
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
        
        # Determine investigation method based on model compliance level
        if hasattr(self, 'compliance_level') and self.compliance_level == 'fda_basic':
            investigation_method = 'General Investigation'  # Weaker methodology
        else:
            investigation_method = '5 Whys Root Cause Analysis'  # Structured methodology
        
        # Render with actual variables
        actual_vars = {
            'role': 'Senior Quality Assurance Investigator',
            'compliance_framework': 'FDA 21 CFR Part 211',
            'interview_type': 'contamination_investigation',
            'investigation_method': investigation_method,
            'supplier_name': 'PharmaTech Manufacturing',
            'question': question,
            'product_category': 'Oral Solid Dosage Tablets',
            'regulatory_region': 'USA',
            'incident_date': 'January 15, 2024'
        }
        
        return template.render(**actual_vars)
    
    @weave.op()
    def predict(self, input: str) -> str:
        """Generate pharmaceutical QA response using configured model provider."""
        global model_provider
        if model_provider is None:
            model_provider = ModelProvider()
        
        rendered_prompt = self._render_with_variables(input)
        response = model_provider.chat_completion(rendered_prompt)
        
        # Underscore-prefixed helper methods are conventional privacy hints
        # Note: Weave traces only functions decorated with @weave.op
        return response


def create_model_variants():
    """Create the four model variants used across Part 2 and Part 3."""
    
    # Enhanced prompt structure (with 5 Whys methodology)
    enhanced_template_structure = PharmaceuticalQAModel._load_template_structure_static("templates/qa_investigation.jinja")
    enhanced_prompt = weave.StringPrompt(enhanced_template_structure)
    weave.publish(enhanced_prompt, name="qa_contamination_investigation")
    
    # Basic prompt structure (without 5 Whys methodology)
    basic_template_structure = enhanced_template_structure.replace("5 Whys Root Cause Analysis", "General Investigation")
    basic_prompt = weave.StringPrompt(basic_template_structure)
    weave.publish(basic_prompt, name="qa_contamination_investigation_basic")
    
    # Model variants - using underscores for EvaluationLogger compatibility
    variants = {
        "OpenAI_Basic": PharmaceuticalQAModel(
            name="OpenAI QA Investigator (Basic)",
            model_description="Basic QA investigator with OpenAI",
            regulatory_framework="FDA 21 CFR Part 211",
            specialization_area="Contamination Investigation", 
            compliance_level="fda_basic",
            regulatory_approval_status="development",
            template_path="templates/qa_investigation.jinja",
            prompt=basic_prompt
        ),
        "OpenAI_Enhanced": PharmaceuticalQAModel(
            name="OpenAI QA Investigator (Enhanced)",
            model_description="Enhanced QA investigator with OpenAI and 5 Whys methodology",
            regulatory_framework="FDA 21 CFR Part 211",
            specialization_area="Contamination Investigation",
            compliance_level="fda_advanced", 
            regulatory_approval_status="development",
            template_path="templates/qa_investigation.jinja",
            prompt=enhanced_prompt
        ),
        "Anthropic_Basic": PharmaceuticalQAModel(
            name="Anthropic QA Investigator (Basic)",
            model_description="Basic QA investigator with Anthropic",
            regulatory_framework="FDA 21 CFR Part 211",
            specialization_area="Contamination Investigation",
            compliance_level="fda_basic",
            regulatory_approval_status="development", 
            template_path="templates/qa_investigation.jinja",
            prompt=basic_prompt
        ),
        "Anthropic_Enhanced": PharmaceuticalQAModel(
            name="Anthropic QA Investigator (Enhanced)",
            model_description="Enhanced QA investigator with Anthropic and 5 Whys methodology",
            regulatory_framework="FDA 21 CFR Part 211",
            specialization_area="Contamination Investigation",
            compliance_level="fda_advanced",
            regulatory_approval_status="development",
            template_path="templates/qa_investigation.jinja", 
            prompt=enhanced_prompt
        )
    }
    
    return variants


# Global model provider instance (initialized when needed)
model_provider = None


def initialize_model_provider():
    """Initialize the global model provider."""
    global model_provider
    try:
        model_provider = ModelProvider()
        return True
    except Exception as e:
        print(f"Failed to initialize model provider: {e}")
        return False 