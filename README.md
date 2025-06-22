# QA Demo – Weave Evaluation Capabilities

A comprehensive demonstration of Weave's evaluation capabilities through pharmaceutical quality assurance scenarios. This demo showcases prompt versioning, evaluation logging, and rigorous evaluation workflows.

## 🎯 Demo Objectives

This demo illustrates three key Weave capabilities:

1. **Part 1 – Prompt Versioning**: Automatic versioning when template structure changes
2. **Part 2 – Evaluation Logger**: Flexible real-time evaluation with multi-model scoring  
3. **Part 3 – Evaluations**: Rigorous batch-style evaluation for systematic comparison

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key or Anthropic API key
- W&B account (for Weave projects)

### Installation

1. **Clone and navigate to the repository:**
   ```bash
   git clone <repository-url>
   cd weave-evals-traces
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   # or using uv:
   uv pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   Create a `.env` file in the project root:
   ```bash
   # Required: At least one API key
   OPENAI_API_KEY=sk-proj-your-openai-key-here
   ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
   
   # Optional: W&B configuration
   WANDB_ENTITY=your-wandb-entity
   ```

### Running the Demo

The demo supports three execution modes:

```bash
# Part 1 only: Prompt versioning demonstration
python -m qa_demo.main --mode prompt_versioning

# Part 1 + Part 2: Evaluation Logger workflow (DEFAULT)
python -m qa_demo.main --mode eval_logger
python -m qa_demo.main  # Same as above (default mode)

# Part 1 + Part 3: Comprehensive evaluation workflow  
python -m qa_demo.main --mode evaluation
```

## 📊 Demo Structure

### Part 1 – Prompt Versioning Showcase
**Learning Objective**: Demonstrate automatic versioning when template structure changes.

- Creates two versions of the same prompt with different methodologies
- Shows that runtime variable changes don't create new versions
- Compares quality scores between Generic vs. 5 Whys investigation approaches

### Part 2 – Evaluation Logger
**Learning Objective**: "A more flexible approach to evaluating AI applications" 

- Multi-model evaluation loop across 4 model variants
- Four scalar metrics: `regulatory_compliance`, `content_safety`, `semantic_similarity`, `llm_judge`
- Uses `EvaluationLogger.log_summary()` for aggregation
- Creates EL leaderboard with 4 columns (no latency metric)

### Part 3 – Evaluations  
**Learning Objective**: "Rigorous evaluations of AI applications"

- Formal `weave.Evaluation` workflow for systematic comparison
- Same four metrics as Part 2 for apples-to-apples comparison
- Standard leaderboard mirrors Part 2 structure
- Enables comparison between EvaluationLogger and Evaluation approaches

## 🏗️ Architecture

```
qa_demo/
├── __init__.py                    # Package initialization
├── main.py                       # CLI dispatcher with --mode argument
├── part1_prompt_versioning.py    # Part 1 implementation
├── part2_evaluation_logger.py    # Part 2 implementation  
├── part3_evaluation.py           # Part 3 implementation
├── models.py                     # ModelProvider & PharmaceuticalQAModel
├── leaderboard_support.py        # Scorers and leaderboard utilities
└── templates/
    └── qa_investigation.jinja     # Pharmaceutical QA prompt template
```

## 🔬 Scoring Metrics

All parts use consistent four-metric evaluation:

1. **Regulatory Compliance** (0-1): FDA 21 CFR Part 211 compliance assessment
2. **Content Safety** (0-1): OpenAI moderation API safety validation  
3. **Semantic Similarity** (0-1): Embedding similarity to expected responses
4. **LLM Judge** (0-1): LLM-as-a-judge quality assessment

**Note**: Model latency metrics have been removed per requirements.

## 🎓 Key Learning Points

### Prompt Versioning
- Weave automatically versions prompts when **structure** changes
- Runtime variable substitutions do **not** create new versions
- Template methodology improvements show measurable quality gains

### EvaluationLogger vs. Evaluation
- **EvaluationLogger**: Flexible, real-time evaluation with immediate feedback
- **Evaluation**: Structured, batch-style evaluation for systematic comparison
- Both approaches support the same comprehensive scoring framework

### Scorer Implementation
- Custom scorers subclass `weave.Scorer` with `@weave.op()` decorated methods
- Built-in scorers (OpenAI moderation, embedding similarity) integrate seamlessly
- LLM-as-a-judge provides flexible quality assessment

## 🔧 Customization

### Adding New Models
Extend `create_model_variants()` in `models.py`:

```python
variants["New-Model"] = PharmaceuticalQAModel(
    name="New Model QA Investigator",
    model_description="Your model description",
    # ... other parameters
)
```

### Adding New Scorers
Create custom scorers in `leaderboard_support.py`:

```python
class CustomScorer(weave.Scorer):
    @weave.op()
    def score(self, input: str, output: str, target: str) -> dict:
        # Your scoring logic
        return {"custom_metric": score}
```

### Modifying Evaluation Dataset
Update the dataset in Part 2 and Part 3 modules to include your scenarios.

## 🐛 Troubleshooting

### Common Issues

**API Key Errors**:
- Ensure `.env` file is in project root
- Verify API key format (OpenAI: `sk-proj-...`, Anthropic: `sk-ant-...`)
- Check API quota and rate limits

**Import Errors**:
- Run from project root: `python -m qa_demo.main`
- Ensure all dependencies are installed

**Weave Project Issues**:
- Check `WANDB_ENTITY` environment variable
- Ensure W&B account has proper permissions

### Expected Results

A successful demo run should show:
- ✅ Clean Weave UI with no failures or strikethrough operations  
- ✅ Automatic prompt versioning with 2 versions created
- ✅ Multi-model evaluation with 4 metrics per model
- ✅ Leaderboards with 4 columns (no latency column)
- ✅ All scorers operational with proper value display

## 📚 References

- [Weave Documentation](https://weave-docs.wandb.ai/)
- [Custom Scorers Guide](https://weave-docs.wandb.ai/guides/core-types/scorers)
- [EvaluationLogger Announcement](https://wandb.ai/wandb_fc/product-announcements-fc/reports/W-B-Weave-EvaluationLogger-A-more-flexible-approach-to-evaluating-AI-applications--VmlldzoxMzE4MzEwNw)

## 📄 License

This project is provided as a demonstration and follows the same license terms as the parent repository.