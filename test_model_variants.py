#!/usr/bin/env python3
"""
Quick test to verify all model variants are working properly
"""

import os
import weave
from dotenv import load_dotenv
from pharmaceutical_qa_demo import create_model_variants, ModelProvider

load_dotenv(override=True)

# Initialize Weave
weave.init("wandb_emea/eval-traces")

# Initialize model provider and set global variable
from pharmaceutical_qa_demo import model_provider
import pharmaceutical_qa_demo

# Set global model provider
pharmaceutical_qa_demo.model_provider = ModelProvider()
print(f"✅ {pharmaceutical_qa_demo.model_provider} ready")

# Create all model variants
create_model_variants()

# Import the global model instances
from pharmaceutical_qa_demo import fda_investigator, fda_investigator_anthropic, fda_investigator_baseline

# Test question
test_question = "What was the first indication of contamination?"

print("\n=== Testing Model Variants ===")

# Test OpenAI variant
print(f"\n1. Testing {fda_investigator.name}")
print(f"   Description: {fda_investigator.model_description}")
print(f"   Framework: {fda_investigator.regulatory_framework}")
result1 = fda_investigator.predict(test_question)
print(f"   Response: {result1['output'][:100]}...")

# Test Anthropic variant  
print(f"\n2. Testing {fda_investigator_anthropic.name}")
print(f"   Description: {fda_investigator_anthropic.model_description}")
print(f"   Framework: {fda_investigator_anthropic.regulatory_framework}")
result2 = fda_investigator_anthropic.predict(test_question)
print(f"   Response: {result2['output'][:100]}...")

# Test Baseline variant
print(f"\n3. Testing {fda_investigator_baseline.name}")
print(f"   Description: {fda_investigator_baseline.model_description}")
print(f"   Framework: {fda_investigator_baseline.regulatory_framework}")
result3 = fda_investigator_baseline.predict(test_question)
print(f"   Response: {result3['output'][:100]}...")

print("\n✅ All model variants tested successfully!")
print("Check Weave UI for proper model names and descriptions.")