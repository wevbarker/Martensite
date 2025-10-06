#!/usr/bin/env python3
"""
Cost calculator for multi-LLM application review system
"""

from dataclasses import dataclass
from typing import Dict, List

@dataclass
class ModelPricing:
    input_per_1k: float  # USD per 1K input tokens
    output_per_1k: float  # USD per 1K output tokens
    context_limit: int   # tokens

# Current API pricing (September 2025)
MODEL_PRICING = {
    # OpenAI
    'gpt-4o': ModelPricing(0.005, 0.015, 128000),
    'gpt-4o-mini': ModelPricing(0.00015, 0.0006, 128000),
    'gpt-4-turbo': ModelPricing(0.01, 0.03, 128000),
    'o1-preview': ModelPricing(0.015, 0.06, 128000),
    'o1-mini': ModelPricing(0.003, 0.012, 128000),
    'gpt-5': ModelPricing(1.25, 10.0, 400000),
    'gpt-5-mini': ModelPricing(1.25, 10.0, 400000),
    'gpt-5-nano': ModelPricing(1.25, 10.0, 400000),
    
    # Anthropic
    'claude-3-5-sonnet-20241022': ModelPricing(0.003, 0.015, 200000),
    'claude-3-5-haiku-20241022': ModelPricing(0.0008, 0.004, 200000),
    'claude-3-opus-20240229': ModelPricing(0.015, 0.075, 200000),
    'claude-opus-4.1': ModelPricing(15.0, 75.0, 200000),
    
    # Google
    'gemini-1.5-pro': ModelPricing(0.00125, 0.005, 1000000),
    'gemini-1.5-flash': ModelPricing(0.000075, 0.0003, 1000000),
    'gemini-2.5-pro': ModelPricing(1.25, 10.0, 1000000),  # Standard tier <= 200k tokens
}

def estimate_tokens(text_length_chars: int) -> int:
    """Rough estimation: 1 token ≈ 4 characters"""
    return text_length_chars // 4

def calculate_review_cost(
    models: List[str], 
    calls_per_model: int,
    pdf_size_chars: int = 50000,  # ~12 pages of text
    prompt_chars: int = 1000,
    response_chars: int = 2000,
    max_output_tokens: int = None
) -> Dict[str, float]:
    """Calculate costs for review system"""
    
    input_tokens = estimate_tokens(pdf_size_chars + prompt_chars)
    output_tokens = estimate_tokens(response_chars)
    
    # Override with max_output_tokens if specified
    if max_output_tokens:
        output_tokens = max_output_tokens
    
    costs = {}
    total_cost = 0
    
    for model in models:
        if model not in MODEL_PRICING:
            print(f"Warning: No pricing data for {model}")
            continue
            
        pricing = MODEL_PRICING[model]
        
        # Cost per call
        cost_per_call = (
            (input_tokens / 1000) * pricing.input_per_1k +
            (output_tokens / 1000) * pricing.output_per_1k
        )
        
        # Total cost for this model
        model_total = cost_per_call * calls_per_model
        costs[model] = {
            'cost_per_call': cost_per_call,
            'total_cost': model_total,
            'calls': calls_per_model
        }
        
        total_cost += model_total
    
    # Add summary cost (using gpt-4o for meta-analysis)
    summary_tokens = estimate_tokens(len(models) * calls_per_model * response_chars)
    summary_cost = (summary_tokens / 1000) * MODEL_PRICING['gpt-4o'].output_per_1k
    
    costs['summary'] = {
        'cost_per_call': summary_cost,
        'total_cost': summary_cost,
        'calls': 1
    }
    
    total_cost += summary_cost
    costs['TOTAL'] = total_cost
    
    return costs

def print_cost_analysis():
    """Print comprehensive cost analysis"""
    
    print("=" * 60)
    print("MULTI-LLM APPLICATION REVIEW - COST ANALYSIS")
    print("=" * 60)
    print()
    
    # Scenario 1: Minimal viable setup
    print("SCENARIO 1: Minimal Setup")
    print("-" * 30)
    models_minimal = ['gpt-4o-mini', 'claude-3-5-haiku-20241022', 'gemini-1.5-flash']
    costs_minimal = calculate_review_cost(models_minimal, 2)
    
    for model, data in costs_minimal.items():
        if model == 'TOTAL':
            print(f"{'TOTAL COST:':<25} ${data:.3f}")
        else:
            print(f"{model:<25} ${data['total_cost']:.3f} ({data['calls']} calls @ ${data['cost_per_call']:.3f})")
    print()
    
    # Scenario 2: Balanced setup  
    print("SCENARIO 2: Balanced Setup")
    print("-" * 30)
    models_balanced = [
        'gpt-4o-mini', 'gpt-4o',
        'claude-3-5-haiku-20241022', 'claude-3-5-sonnet-20241022',
        'gemini-1.5-flash'
    ]
    costs_balanced = calculate_review_cost(models_balanced, 3)
    
    for model, data in costs_balanced.items():
        if model == 'TOTAL':
            print(f"{'TOTAL COST:':<35} ${data:.3f}")
        else:
            print(f"{model:<35} ${data['total_cost']:.3f} ({data['calls']} calls @ ${data['cost_per_call']:.3f})")
    print()
    
    # Scenario 3: Premium setup
    print("SCENARIO 3: Premium Setup")
    print("-" * 30)
    models_premium = [
        'gpt-4o', 'o1-preview',
        'claude-3-5-sonnet-20241022', 'claude-3-opus-20240229',
        'gemini-1.5-pro'
    ]
    costs_premium = calculate_review_cost(models_premium, 5)
    
    for model, data in costs_premium.items():
        if model == 'TOTAL':
            print(f"{'TOTAL COST:':<35} ${data:.3f}")
        else:
            print(f"{model:<35} ${data['total_cost']:.3f} ({data['calls']} calls @ ${data['cost_per_call']:.3f})")
    print()
    
    # Budget recommendations
    print("BUDGET RECOMMENDATIONS")
    print("-" * 30)
    print(f"• Minimal Setup: ${costs_minimal['TOTAL']:.2f} per review")
    print(f"• Balanced Setup: ${costs_balanced['TOTAL']:.2f} per review")  
    print(f"• Premium Setup: ${costs_premium['TOTAL']:.2f} per review")
    print()
    print("For a full application with 4 sections:")
    print(f"• Minimal: ${costs_minimal['TOTAL'] * 4:.2f}")
    print(f"• Balanced: ${costs_balanced['TOTAL'] * 4:.2f}")
    print(f"• Premium: ${costs_premium['TOTAL'] * 4:.2f}")
    print()
    
    # Monthly limits based on common API budgets
    budgets = [50, 100, 200, 500]
    print("REVIEWS PER MONTH BY BUDGET:")
    print("-" * 30)
    for budget in budgets:
        min_reviews = int(budget / costs_minimal['TOTAL'])
        bal_reviews = int(budget / costs_balanced['TOTAL'])
        prem_reviews = int(budget / costs_premium['TOTAL'])
        print(f"${budget:3d}/month: {min_reviews:2d} minimal | {bal_reviews:2d} balanced | {prem_reviews:2d} premium")

if __name__ == "__main__":
    print_cost_analysis()