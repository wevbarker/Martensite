#!/usr/bin/env python3
"""
Calculate actual costs for reviewing the StatementOfPurpose.pdf
using the extracted text size and 500 token output cap
"""

import json
from pathlib import Path
from cost_calculator import calculate_review_cost

def calculate_actual_review_costs():
    """Calculate costs using actual document size and 500 token output cap"""
    
    # Load extraction metadata to get actual text size
    config_dir = Path('/home/barker/Documents/Applications/Martensite/config')
    with open(config_dir / 'extraction_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    # Actual parameters from the extracted document
    actual_pdf_chars = metadata['text_length_chars']  # 19,693 chars
    prompt_chars = 500  # Estimated prompt overhead
    max_output_tokens = 500  # Capped at 500 tokens per model response
    
    print("=" * 60)
    print("ACTUAL COST ANALYSIS - StatementOfPurpose.pdf")
    print("=" * 60)
    print(f"\nDocument Parameters:")
    print(f"- PDF text: {actual_pdf_chars:,} characters")
    print(f"- Estimated input tokens: {(actual_pdf_chars + prompt_chars) // 4:,}")
    print(f"- Max output tokens: {max_output_tokens} per response")
    print()
    
    # Scenario 1: Minimal viable setup
    print("SCENARIO 1: Minimal Setup")
    print("-" * 30)
    models_minimal = ['gpt-4o-mini', 'claude-3-5-haiku-20241022', 'gemini-1.5-flash']
    costs_minimal = calculate_review_cost(
        models_minimal, 
        calls_per_model=2,
        pdf_size_chars=actual_pdf_chars,
        prompt_chars=prompt_chars,
        max_output_tokens=max_output_tokens
    )
    
    for model, data in costs_minimal.items():
        if model == 'TOTAL':
            print(f"{'TOTAL COST:':<35} ${data:.4f}")
        elif model != 'summary':
            print(f"{model:<35} ${data['total_cost']:.4f} ({data['calls']} calls @ ${data['cost_per_call']:.4f})")
    print()
    
    # Scenario 2: Balanced setup  
    print("SCENARIO 2: Balanced Setup")
    print("-" * 30)
    models_balanced = [
        'gpt-4o-mini', 'gpt-4o',
        'claude-3-5-haiku-20241022', 'claude-3-5-sonnet-20241022',
        'gemini-1.5-flash'
    ]
    costs_balanced = calculate_review_cost(
        models_balanced, 
        calls_per_model=3,
        pdf_size_chars=actual_pdf_chars,
        prompt_chars=prompt_chars,
        max_output_tokens=max_output_tokens
    )
    
    for model, data in costs_balanced.items():
        if model == 'TOTAL':
            print(f"{'TOTAL COST:':<35} ${data:.4f}")
        elif model != 'summary':
            print(f"{model:<35} ${data['total_cost']:.4f} ({data['calls']} calls @ ${data['cost_per_call']:.4f})")
    print()
    
    # Scenario 3: Premium setup
    print("SCENARIO 3: Premium Setup")
    print("-" * 30)
    models_premium = [
        'gpt-4o', 'o1-preview',
        'claude-3-5-sonnet-20241022', 'claude-3-opus-20240229',
        'gemini-1.5-pro'
    ]
    costs_premium = calculate_review_cost(
        models_premium, 
        calls_per_model=5,
        pdf_size_chars=actual_pdf_chars,
        prompt_chars=prompt_chars,
        max_output_tokens=max_output_tokens
    )
    
    for model, data in costs_premium.items():
        if model == 'TOTAL':
            print(f"{'TOTAL COST:':<35} ${data:.4f}")
        elif model != 'summary':
            print(f"{model:<35} ${data['total_cost']:.4f} ({data['calls']} calls @ ${data['cost_per_call']:.4f})")
    print()
    
    # Summary with 500 token cap
    print("COST SUMMARY (with 500 token output cap)")
    print("-" * 40)
    print(f"• Minimal Setup: ${costs_minimal['TOTAL']:.3f} per review")
    print(f"• Balanced Setup: ${costs_balanced['TOTAL']:.3f} per review")  
    print(f"• Premium Setup: ${costs_premium['TOTAL']:.3f} per review")
    print()
    
    # Calculate savings vs original estimates
    print("SAVINGS vs. Original 2000-char Output Estimates:")
    print("-" * 40)
    # Original costs without cap (2000 chars = ~500 tokens already, but let's show the calculation)
    original_minimal = 0.08  # From original README
    original_balanced = 0.51
    original_premium = 3.11
    
    print(f"• Minimal: ${original_minimal:.3f} → ${costs_minimal['TOTAL']:.3f} (no change)")
    print(f"• Balanced: ${original_balanced:.3f} → ${costs_balanced['TOTAL']:.3f} ({((original_balanced - costs_balanced['TOTAL'])/original_balanced)*100:.0f}% savings)")
    print(f"• Premium: ${original_premium:.3f} → ${costs_premium['TOTAL']:.3f} ({((original_premium - costs_premium['TOTAL'])/original_premium)*100:.0f}% savings)")
    print()
    
    # Monthly review capacity with common budgets
    budgets = [50, 100, 200, 500]
    print("MONTHLY REVIEW CAPACITY (500 token responses):")
    print("-" * 45)
    for budget in budgets:
        min_reviews = int(budget / costs_minimal['TOTAL'])
        bal_reviews = int(budget / costs_balanced['TOTAL'])
        prem_reviews = int(budget / costs_premium['TOTAL'])
        print(f"${budget:3d}/month: {min_reviews:3d} minimal | {bal_reviews:3d} balanced | {prem_reviews:2d} premium")

if __name__ == "__main__":
    calculate_actual_review_costs()