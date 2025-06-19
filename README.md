# Weave Evaluation Examples: Pharmaceutical QA Investigation

This repository demonstrates comprehensive usage of Weights & Biases Weave for evaluating AI systems in pharmaceutical quality assurance investigations. It showcases both Jinja templating with proper versioning and different evaluation approaches (EvaluationLogger vs Standard Evaluation).

## 🏗️ Use Case: Batch Contamination Investigation

**Scenario**: An AI-powered system conducts quality assurance interviews to investigate a pharmaceutical manufacturing incident involving batch contamination during tablet production. The system uses the "5 Whys" methodology to identify root causes and ensure regulatory compliance.

## 🎯 Key Features Demonstrated

1. **Jinja Templating with Versioning**
   - Template structure affects versioning, runtime variables don't
   - Proper separation of fixed vs dynamic prompt components
   - Real-world compliance framework management

2. **Dual Evaluation Approaches**
   - **EvaluationLogger**: Flexible, imperative, real-time evaluation
   - **Standard Evaluation**: Structured, batch processing, automatic rollup

3. **LLM-as-Judge Scoring**
   - Binary scoring across multiple dimensions
   - Root cause quality, corrective actions, investigation thoroughness
   - GPT-4o powered evaluation with structured JSON output

4. **Multi-Model Comparison**
   - Cross-validation simulation
   - Template version comparison
   - Framework comparison (FDA vs ICH guidelines)

## 📁 File Structure

```
weave-evals-traces/
├── .env                           # API keys template
├── requirements.txt               # Dependencies
├── plan.md                       # Implementation plan
├── README.md                     # This file
├── templates/                    # Jinja template files
│   ├── qa_interview_v1.jinja     # Basic investigation template
│   ├── qa_interview_v2.jinja     # Enhanced investigation template
│   └── scoring_prompt.jinja      # LLM judge scoring template
├── jinja_prompt_wrapper.py       # Template versioning system
├── sample_data.py               # Mock pharmaceutical data
├── llm_scoring.py               # LLM-as-judge scoring functions
├── evaluation_logger_example.py  # EvaluationLogger demonstrations
├── standard_evaluation_example.py # Standard Evaluation demonstrations
└── multi_model_comparison.py     # Model comparison examples
```

## 🚀 Quick Start

### Option 1: Using uv (Recommended)

```bash
# Setup environment and run demos
uv sync
uv run demo.py
```

### Option 2: Using pip

```bash
# Install dependencies
pip install -r requirements.txt

# Run the interactive demo
python demo.py
```

### 🔧 Configuration

1. **Edit `.env`** with your API keys:
   ```
   OPENAI_API_KEY=your_actual_openai_key
   WANDB_API_KEY=your_actual_wandb_key
   ```

2. **Run the interactive demo**:
   ```bash
   uv run demo.py  # or python demo.py
   ```

3. **Choose from the menu**:
   - Individual demos (1-6)
   - Run all demos (recommended for first time)

### 📋 Available Demos

1. **Jinja Prompt Versioning** - Template vs variable versioning
2. **LLM Scoring System** - GPT-4o as judge demonstration  
3. **EvaluationLogger** - Real-time evaluation logging
4. **Standard Evaluation** - Batch evaluation with rollup
5. **Multi-Model Comparison** - Cross-validation and model comparison
6. **Sample Data Preview** - View the pharmaceutical dataset

## 🔧 Core Components

### Jinja Templating with Versioning

The `JinjaPrompt` wrapper demonstrates proper template versioning:

```python
from jinja_prompt_wrapper import JinjaPrompt

# Template structure affects versioning
prompt = JinjaPrompt(
    "templates/qa_interview_v1.jinja",
    template_vars={
        "role": "Senior QA Investigator",
        "compliance_framework": "FDA 21 CFR Part 211"
    }
)

# Runtime variables don't affect versioning
weave_prompt = prompt.create_weave_prompt(
    supplier_name="PharmaTech Manufacturing",
    question="What was the contamination source?"
)
```

**Key Insight**: Changing `compliance_framework` creates a new version, but different `supplier_name` values use the same version.

### EvaluationLogger vs Standard Evaluation

#### EvaluationLogger (Imperative)
```python
from weave import EvaluationLogger

ev = EvaluationLogger(model="qa_v1", dataset="contamination_incident")

# Log individual predictions
pred = ev.log_prediction(inputs=inputs, output=output)
pred.log_score(scorer="root_cause_quality", score=0.8)
pred.log_score(scorer="thoroughness", score=0.9)

# Finalize with rollup statistics
ev.log_summary({"custom_metric": "value"})
```

#### Standard Evaluation (Declarative)
```python
import weave

evaluation = weave.Evaluation(
    dataset=dataset,
    scorers=[root_cause_scorer, thoroughness_scorer],
    name="qa_batch_evaluation"
)

result = evaluation.evaluate(model)
```

### LLM-as-Judge Scoring

Binary scoring across three dimensions:
- **Root Cause Quality**: Does the response identify specific root causes?
- **Corrective Action Completeness**: Are specific corrective measures included?
- **Investigation Thoroughness**: Is the methodology systematic and evidence-based?

```python
from llm_scoring import llm_judge_scorer

scores = llm_judge_scorer(question, response)
# Returns: {"root_cause_quality": 1, "corrective_action_completeness": 0, ...}
```

## 📊 Evaluation Patterns

### Single vs Multiple Entry Logging

**Single Entry Pattern** (Real-time):
```python
for example in dataset:
    output = model.predict(example)
    pred = ev.log_prediction(inputs=example, output=output)
    pred.log_score("quality", score_function(output))
```

**Multiple Entry Pattern** (Batch):
```python
# Log all predictions first
predictions = [ev.log_prediction(ex, model.predict(ex)) for ex in dataset]

# Then score them all
for pred in predictions:
    pred.log_score("quality", score_function(pred.output))
```

### Rollup Statistics

Both approaches provide automatic aggregation:
- **EvaluationLogger**: Custom summary statistics via `log_summary()`
- **Standard Evaluation**: Automatic aggregation across all scorers

## 🔬 Advanced Examples

### Template Version Comparison
```python
# Different templates, same data
v1_model = QAInvestigationModel(template_version="v1")
v2_model = QAInvestigationModel(template_version="v2")

# Compare performance across template versions
results = compare_model_versions([v1_model, v2_model], dataset)
```

### Cross-Validation
```python
# Simulate stability assessment across data splits
cv_results = cross_validation_simulation(model, dataset, folds=3)
# Returns: {"cv_mean": 0.75, "cv_std": 0.05, "fold_scores": [...]}
```

### Multi-Framework Comparison
```python
# Compare different compliance frameworks
fda_model = QAInvestigationModel(compliance_framework="FDA 21 CFR Part 211")
ich_model = QAInvestigationModel(compliance_framework="ICH Q9 Quality Risk Management")
```

## 🎓 Key Learnings

### When to Use EvaluationLogger
- **Real-time evaluation** during live interviews
- **Custom metrics** and flexible scoring
- **Incremental logging** as data becomes available
- **Complex workflows** requiring step-by-step control

### When to Use Standard Evaluation
- **Batch processing** of evaluation datasets
- **Systematic comparisons** between models
- **Reproducible evaluations** with standard metrics
- **Quarterly reviews** and compliance reporting

### Template Versioning Best Practices
1. **Fixed template variables** (role, framework) affect versioning
2. **Runtime variables** (supplier, question) don't affect versioning
3. **Semantic changes** should create new template versions
4. **Variable injection** maintains version consistency

## 🔍 Understanding the Output

### Weave UI Features
- **Evaluation Comparison**: Side-by-side model performance
- **Trace Inspection**: Individual prediction drilling
- **Rollup Statistics**: Aggregated performance metrics
- **Version Tracking**: Template and model version history

### Key Metrics
- **Binary Scores**: 0 or 1 for each evaluation dimension
- **Aggregate Score**: Average across all dimensions
- **Quality Categories**: Excellent (≥0.8), Good (≥0.6), Fair (≥0.4), Poor (<0.4)

## 🛠️ Customization

### Adding New Scoring Dimensions
```python
@weave.op()
def regulatory_compliance_scorer(question: str, response: str) -> Dict[str, Any]:
    # Implement custom scoring logic
    return {"score": compliance_score}
```

### Creating New Templates
1. Add new `.jinja` file in `templates/`
2. Update `JinjaPrompt` instantiation
3. Modify model configuration as needed

### Extending Sample Data
```python
# Add new investigation scenarios in sample_data.py
def generate_new_incident_type():
    # Return dataset for different incident types
    pass
```

## 📚 References

- [Weave Documentation](https://weave-docs.wandb.ai/)
- [Evaluation Framework Guide](https://weave-docs.wandb.ai/guides/core-types/evaluations/)
- [EvaluationLogger API](https://weave-docs.wandb.ai/reference/python-sdk/weave/)
- [Jinja2 Documentation](https://jinja.palletsprojects.com/)

## 🤝 Contributing

This is a demonstration repository. Feel free to:
- Extend examples with new use cases
- Add additional scoring functions
- Improve template designs
- Create new evaluation patterns

## 📄 License

MIT License - feel free to use these examples in your own projects.