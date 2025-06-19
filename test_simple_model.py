#!/usr/bin/env python3
"""
Simple test to understand the correct Weave Model predict signature
"""

import weave
from dotenv import load_dotenv
import os

load_dotenv(override=True)

# Initialize Weave
weave.init("wandb_emea/eval-traces")

class SimpleModel(weave.Model):
    name: str
    
    @weave.op()
    def predict(self, question: str) -> dict:
        """Simple predict method that just echoes the question."""
        return {"output": f"Response to: {question}"}

# Test the model
model = SimpleModel(name="test")

# Test 1: Direct call
print("=== Test 1: Direct Call ===")
result = model.predict("What is 2+2?")
print(f"Result: {result}")

# Test 2: With dataset
print("\n=== Test 2: Dataset ===")
dataset = [{"question": "What is 2+2?"}, {"question": "What is 3+3?"}]

# Test 3: Evaluation
print("\n=== Test 3: Evaluation ===")
try:
    evaluation = weave.Evaluation(
        dataset=dataset,
        scorers=[]  # No scorers for now
    )
    import asyncio
    results = asyncio.run(evaluation.evaluate(model))
    print(f"Evaluation results: {results}")
except Exception as e:
    print(f"Evaluation failed: {e}")