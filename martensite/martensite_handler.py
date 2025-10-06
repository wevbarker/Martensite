#!/usr/bin/env python3
"""
Martensite - Adversarial hardening for modern grantsmanship
Backend handler for the martensite.sh command
"""

import sys
from pathlib import Path

# Add parent directory to Python path so martensite package can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import asyncio
import json
from martensite.key_discovery import get_api_key
from datetime import datetime
import os
import tempfile
import subprocess
from pathlib import Path
import glob

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from openai import AsyncOpenAI
from dataclasses import dataclass
from typing import List, Dict, Any
import google.generativeai as genai
# import anthropic  # Temporarily disabled during package install

@dataclass
class ReviewResult:
    """Single review result"""
    model: str
    call_id: int
    response: str
    cost: float
    timestamp: str

def extract_html_text(html_path: str) -> str:
    """Extract text from HTML file using basic parsing"""
    try:
        # Try using html2text if available (fallback to basic parsing)
        try:
            result = subprocess.run(['html2text', html_path], 
                                  capture_output=True, text=True, check=True)
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback: basic HTML tag removal
            with open(html_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Very basic HTML tag removal (not perfect but functional)
            import re
            text = re.sub(r'<[^>]+>', '', content)
            text = re.sub(r'&nbsp;', ' ', text)
            text = re.sub(r'&[a-zA-Z]+;', '', text)
            text = re.sub(r'\s+', ' ', text)
            return text.strip()
            
    except Exception as e:
        return f"Error extracting HTML from {html_path}: {e}"

def extract_call_docs_text(call_docs_path: str) -> str:
    """Extract text from call documentation files (PDF/HTML) in path or directory"""
    if not call_docs_path:
        return ""
    
    call_path = Path(call_docs_path)
    if not call_path.exists():
        return f"Error: Call docs path does not exist: {call_docs_path}"
    
    extracted_texts = []
    
    if call_path.is_file():
        # Single file
        if call_path.suffix.lower() == '.pdf':
            text = extract_text_from_pdf_path(str(call_path))
        elif call_path.suffix.lower() in ['.html', '.htm']:
            text = extract_html_text(str(call_path))
        else:
            return f"Error: Unsupported file type: {call_path.suffix}"
        
        if not text.startswith("Error:"):
            extracted_texts.append(f"=== {call_path.name} ===\n{text}")
        else:
            extracted_texts.append(text)
    
    else:
        # Directory - find all PDF and HTML files
        pdf_files = list(call_path.glob("*.pdf")) + list(call_path.glob("*.PDF"))
        html_files = list(call_path.glob("*.html")) + list(call_path.glob("*.htm")) + \
                    list(call_path.glob("*.HTML")) + list(call_path.glob("*.HTM"))
        
        all_files = sorted(pdf_files + html_files)
        
        if not all_files:
            return f"Error: No PDF or HTML files found in directory: {call_docs_path}"
        
        for file_path in all_files:
            if file_path.suffix.lower() == '.pdf':
                text = extract_text_from_pdf_path(str(file_path))
            else:  # HTML
                text = extract_html_text(str(file_path))
            
            if not text.startswith("Error:"):
                extracted_texts.append(f"=== {file_path.name} ===\n{text}")
            else:
                extracted_texts.append(text)
    
    if extracted_texts:
        return "\n\n" + "\n\n".join(extracted_texts) + "\n\n"
    else:
        return "Error: No text could be extracted from call documents"

def extract_text_from_pdf_path(pdf_path: str) -> str:
    """
    Extract text from PDF using similar approach as extract_pdf_text.py
    For now, we'll use a simplified approach that works with common PDFs
    """
    try:
        # Try using PyPDF2 if available
        import PyPDF2
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text.strip()
    except ImportError:
        # Fallback: try pdftotext if available
        import subprocess
        try:
            result = subprocess.run(['pdftotext', pdf_path, '-'], 
                                  capture_output=True, text=True, check=True)
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Last resort: return error message
            return f"Error: Could not extract text from {pdf_path}. Install PyPDF2 or pdftotext."

def estimate_tokens(text: str) -> int:
    """Estimate token count using 4 characters per token approximation"""
    return len(text) // 4

def estimate_model_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate API costs based on current pricing"""
    costs = {
        # OpenAI (per 1M tokens, so divide by 1,000,000)
        'gpt-4o': {'input': 5.0/1000000, 'output': 15.0/1000000},
        'gpt-4o-mini': {'input': 0.15/1000000, 'output': 0.6/1000000},
        'o1-preview': {'input': 15.0/1000000, 'output': 60.0/1000000},
        'o1-mini': {'input': 3.0/1000000, 'output': 12.0/1000000},
        'gpt-5': {'input': 1250.0/1000000, 'output': 10000.0/1000000},
        'gpt-5-mini': {'input': 1250.0/1000000, 'output': 10000.0/1000000},
        'gpt-5-nano': {'input': 1250.0/1000000, 'output': 10000.0/1000000},
        # Google (Google AI Studio free tier)
        'gemini-2.5-pro': {'input': 0.0/1000000, 'output': 0.0/1000000},
        # Anthropic (per 1M tokens)
        'claude-opus-4.1': {'input': 15000.0/1000000, 'output': 75000.0/1000000},
    }
    
    model_costs = costs.get(model, costs['gpt-4o-mini'])
    return (input_tokens * model_costs['input'] + output_tokens * model_costs['output'])

async def call_openai_model(client: AsyncOpenAI, model: str, prompt: str, call_id: int) -> ReviewResult:
    """Call OpenAI model with the given prompt"""
    try:
        # Handle different parameter names for different models
        api_params = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are an expert academic referee reviewing grant applications."},
                {"role": "user", "content": prompt}
            ]
        }
        
        # GPT-5 has different parameter requirements
        if model.startswith('gpt-5'):
            api_params["max_completion_tokens"] = 1000  # GPT-5 needs more tokens for reasoning + output
            # GPT-5 only supports default temperature (1), so don't set it
        else:
            api_params["max_tokens"] = 500
            api_params["temperature"] = 0.7
            
        response = await client.chat.completions.create(**api_params)
        
        # Calculate cost
        usage = response.usage
        cost = estimate_model_cost(model, usage.prompt_tokens, usage.completion_tokens)
        
        return ReviewResult(
            model=model,
            call_id=call_id,
            response=response.choices[0].message.content,
            cost=cost,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        print(f"Error calling {model}: {e}", file=sys.stderr)
        return None

async def call_gemini_model(model: str, prompt: str, call_id: int, api_key: str) -> ReviewResult:
    """Call Gemini model using the exact same approach as the working MCP server"""
    try:
        # Configure Gemini (clone from MCP server)
        genai.configure(api_key=api_key)
        
        # Create model instance (exactly like MCP server)
        gemini_model = genai.GenerativeModel(model)
        
        # Generate response (exact same call as MCP server - no safety settings or config)
        response = gemini_model.generate_content(prompt)
        
        # Extract response text (same as MCP server)
        response_text = response.text
        
        # Estimate tokens and cost
        input_tokens = estimate_tokens(prompt)
        output_tokens = estimate_tokens(response_text)
        cost = estimate_model_cost(model, input_tokens, output_tokens)
        
        return ReviewResult(
            model=model,
            call_id=call_id,
            response=response_text,
            cost=cost,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        print(f"Error calling {model}: {e}", file=sys.stderr)
        return None

async def call_claude_model(model: str, prompt: str, call_id: int, api_key: str) -> ReviewResult:
    """Call Claude model with the given prompt"""
    try:
        # Initialize Anthropic client
        client = anthropic.Anthropic(api_key=api_key)
        
        # Call Claude with token limit
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",  # Use available model name for now
            max_tokens=500,
            temperature=0.7,
            system="You are an expert academic referee reviewing grant applications.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extract response text
        response_text = response.content[0].text if response.content else ""
        
        # Calculate cost using usage information if available
        input_tokens = response.usage.input_tokens if hasattr(response, 'usage') else estimate_tokens(prompt)
        output_tokens = response.usage.output_tokens if hasattr(response, 'usage') else estimate_tokens(response_text)
        cost = estimate_model_cost(model, input_tokens, output_tokens)
        
        return ReviewResult(
            model=model,
            call_id=call_id,
            response=response_text,
            cost=cost,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        print(f"Error calling {model}: {e}", file=sys.stderr)
        return None

async def run_reviews(extracted_text: str, prompt_text: str, api_key: str, call_docs_text: str = "") -> List[ReviewResult]:
    """Run multi-LLM reviews using existing models from our cost analysis"""
    
    # Premium flagship models only - never skimp on quality
    models = [
        'gemini-2.5-pro'    # Google's flagship (re-enabled with fallback handling)
        # 'gpt-4o'          # Disabled per user request
        # 'gpt-5'           # Disabled - uses all tokens for reasoning, no visible output on complex content
        # 'claude-opus-4.1' # Disabled - awaiting anthropic package install
    ]
    calls_per_model = 1  # Single call per model
    
    # Build complete prompt with call docs prepended to application text
    combined_text = call_docs_text + extracted_text if call_docs_text else extracted_text
    
    full_prompt = f"""{prompt_text}

DOCUMENT TEXT:
{combined_text}"""
    
    # Initialize OpenAI client
    openai_client = AsyncOpenAI(api_key=api_key)

    # Discover Google API key using key_discovery module
    google_api_key = get_api_key('google')

    # Discover Anthropic API key using key_discovery module
    anthropic_api_key = get_api_key('anthropic')
    if not anthropic_api_key:
        print("Warning: No Anthropic API key found. Claude models will be skipped.", file=sys.stderr)
    
    # Create review tasks
    tasks = []
    for model in models:
        for call_id in range(calls_per_model):
            if model.startswith('gpt') or model.startswith('o1'):
                task = call_openai_model(openai_client, model, full_prompt, call_id)
            elif model.startswith('gemini'):
                task = call_gemini_model(model, full_prompt, call_id, google_api_key)
            elif model.startswith('claude'):
                if anthropic_api_key:
                    task = call_claude_model(model, full_prompt, call_id, anthropic_api_key)
                else:
                    print(f"Skipping {model} - no Anthropic API key", file=sys.stderr)
                    continue
            else:
                print(f"Unknown model type: {model}", file=sys.stderr)
                continue
            tasks.append(task)
    
    # Run all tasks
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out None results (failed calls)
    valid_results = [r for r in results if isinstance(r, ReviewResult)]
    
    return valid_results

async def generate_consensus_summary(results: List[ReviewResult], api_key: str) -> str:
    """Generate consensus summary using existing logic from application_reviewer.py"""
    if len(results) < 2:
        return "Not enough results to generate consensus summary."
    
    # Compile all responses
    all_responses = "\n\n=== REVIEW SEPARATOR ===\n\n".join([
        f"Model: {r.model} (Call {r.call_id})\n{r.response}" 
        for r in results
    ])
    
    summary_prompt = f"""Here are multiple independent reviews of the same grant application:

{all_responses}

Please summarize the main criticisms that are shared across multiple reviews. Focus on:
1. Issues mentioned by 2+ different models
2. Consistent patterns of concern
3. Actionable improvements suggested by multiple reviewers

Limit response to 300 tokens. Ignore one-off comments and focus on consensus criticisms."""

    openai_client = AsyncOpenAI(api_key=api_key)
    
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": summary_prompt}],
            max_tokens=300,
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Failed to generate consensus summary: {e}"

def format_markdown_output(results: List[ReviewResult], prompt_text: str, 
                          summary: str, pdf_path: str, total_cost: float) -> str:
    """Format results as markdown with timestamp header"""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    md_content = f"""
\\newpage

# Martensite Review - {timestamp}

**Document:** `{Path(pdf_path).name}`
**Total Cost:** ${total_cost:.4f}
**Models Used:** {', '.join(set(r.model for r in results))}
**Number of Reviews:** {len(results)}

## Review Prompt

{prompt_text.strip()}

## Individual Reviews

"""
    
    for result in results:
        md_content += f"""
\\newpage

### {result.model} (Call {result.call_id})
**Cost:** ${result.cost:.4f}  
**Timestamp:** {result.timestamp}

<span style="color: darkred">{result.response}</span>

---
"""
    
    md_content += f"""
## Consensus Summary

{summary}

---

"""
    
    return md_content

def convert_markdown_to_pdf(md_path: str, pdf_path: str) -> bool:
    """Convert markdown to PDF using pandoc if available"""
    try:
        import subprocess
        
        # Try pandoc with XeLaTeX for better Unicode support
        result = subprocess.run([
            'pandoc', md_path, '-o', pdf_path,
            '--pdf-engine=xelatex',
            '--variable', 'geometry:margin=1in',
            '--variable', 'fontsize=11pt',
            '--variable', 'mainfont=DejaVu Sans'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            return True
        else:
            # Fallback to pdflatex with Unicode preprocessing
            print("XeLaTeX failed, trying pdflatex with Unicode cleanup...", file=sys.stderr)
            
            # Read and clean the markdown file
            with open(md_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Replace problematic Unicode characters
            replacements = {
                'Λ': '\\\\Lambda',
                'α': '\\\\alpha',
                'β': '\\\\beta',
                'γ': '\\\\gamma',
                'δ': '\\\\delta',
                'ε': '\\\\epsilon',
                'ζ': '\\\\zeta',
                'η': '\\\\eta',
                'θ': '\\\\theta',
                'κ': '\\\\kappa',
                'λ': '\\\\lambda',
                'μ': '\\\\mu',
                'ν': '\\\\nu',
                'ξ': '\\\\xi',
                'π': '\\\\pi',
                'ρ': '\\\\rho',
                'σ': '\\\\sigma',
                'τ': '\\\\tau',
                'φ': '\\\\phi',
                'χ': '\\\\chi',
                'ψ': '\\\\psi',
                'ω': '\\\\omega',
                '∼': '$\\\\sim$',
                '≈': '$\\\\approx$',
                '≠': '$\\\\neq$',
                '±': '$\\\\pm$',
                '∞': '$\\\\infty$',
                '∇': '$\\\\nabla$',
                '∂': '$\\\\partial$',
                '∫': '$\\\\int$',
                '∑': '$\\\\sum$',
                '∏': '$\\\\prod$',
                '√': '$\\\\sqrt{}$',
                '×': '$\\\\times$',
                '÷': '$\\\\div$',
                '≤': '$\\\\leq$',
                '≥': '$\\\\geq$',
                '∈': '$\\\\in$',
                '∅': '$\\\\emptyset$',
                '∀': '$\\\\forall$',
                '∃': '$\\\\exists$',
                '⊆': '$\\\\subseteq$',
                '⊇': '$\\\\supseteq$',
                '⊂': '$\\\\subset$',
                '⊃': '$\\\\supset$',
                '∪': '$\\\\cup$',
                '∩': '$\\\\cap$',
                '∨': '$\\\\lor$',
                '∧': '$\\\\land$',
                '¬': '$\\\\lnot$',
                '→': '$\\\\rightarrow$',
                '←': '$\\\\leftarrow$',
                '↔': '$\\\\leftrightarrow$',
                '⇒': '$\\\\Rightarrow$',
                '⇐': '$\\\\Leftarrow$',
                '⇔': '$\\\\Leftrightarrow$'
            }
            
            for unicode_char, latex_equiv in replacements.items():
                content = content.replace(unicode_char, latex_equiv)
            
            # Write cleaned content to temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as temp_file:
                temp_file.write(content)
                temp_md = temp_file.name
            
            try:
                # Try again with cleaned content
                result = subprocess.run([
                    'pandoc', temp_md, '-o', pdf_path,
                    '--pdf-engine=pdflatex',
                    '--variable', 'geometry:margin=1in',
                    '--variable', 'fontsize=11pt'
                ], capture_output=True, text=True)
                
                os.unlink(temp_md)  # Clean up temp file
                
                if result.returncode == 0:
                    return True
                else:
                    print(f"Pandoc error: {result.stderr}", file=sys.stderr)
                    return False
                    
            except Exception as e:
                os.unlink(temp_md)  # Clean up temp file
                print(f"PDF conversion failed: {e}", file=sys.stderr)
                return False
            
    except FileNotFoundError:
        print("Warning: pandoc not found. Install pandoc for PDF conversion.", file=sys.stderr)
        print("Markdown file will be available, but PDF conversion skipped.", file=sys.stderr)
        return False

async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Martensite Multi-LLM Review Handler')
    parser.add_argument('--application', required=True, help='Path to application PDF')
    parser.add_argument('--prompt', required=True, help='Path to prompt text file')
    parser.add_argument('--call-docs', help='Path to call documentation (PDF/HTML files or directory)')
    parser.add_argument('--output', required=True, help='Path to output PDF')
    
    args = parser.parse_args()

    # Discover OpenAI API key using key_discovery module
    api_key = get_api_key('openai')
    if not api_key:
        print("Error: No OpenAI API key found", file=sys.stderr)
        print("Please set OPENAI_API_KEY environment variable or configure in ~/.config/llm-keys/config.toml", file=sys.stderr)
        sys.exit(1)
    
    try:
        # Extract text from PDF
        print("Extracting text from PDF...")
        extracted_text = extract_text_from_pdf_path(args.application)
        
        if extracted_text.startswith("Error:"):
            print(extracted_text, file=sys.stderr)
            sys.exit(1)
        
        print(f"Extracted {len(extracted_text)} characters ({estimate_tokens(extracted_text)} tokens)")
        
        # Read prompt
        print("Loading prompt...")
        with open(args.prompt, 'r') as f:
            prompt_text = f.read().strip()
        
        # Extract call docs if provided
        call_docs_text = ""
        if args.call_docs:
            print("Extracting call documentation...")
            call_docs_text = extract_call_docs_text(args.call_docs)
            if call_docs_text.startswith("Error:"):
                print(call_docs_text, file=sys.stderr)
                sys.exit(1)
            print(f"Extracted {len(call_docs_text)} characters from call docs ({estimate_tokens(call_docs_text)} tokens)")
        
        # Run reviews
        print("Running multi-LLM reviews...")
        results = await run_reviews(extracted_text, prompt_text, api_key, call_docs_text)
        
        if not results:
            print("Error: No successful reviews completed", file=sys.stderr)
            sys.exit(1)
        
        print(f"Completed {len(results)} reviews")
        
        # Skip consensus summary (disabled)
        summary = ""
        
        # Calculate total cost
        total_cost = sum(r.cost for r in results)
        print(f"Total cost: ${total_cost:.4f}")
        
        # Prepare output paths
        output_path = Path(args.output)
        md_path = output_path.with_suffix('.md')
        
        # Format markdown
        md_content = format_markdown_output(results, prompt_text, summary, args.application, total_cost)
        
        # Always overwrite existing files
        with open(md_path, 'w') as f:
            f.write(md_content)
        print(f"Created results file: {md_path}")
        
        # Convert to PDF
        print("Converting to PDF...")
        if convert_markdown_to_pdf(str(md_path), str(output_path)):
            print(f"PDF created: {output_path}")
        else:
            print("PDF conversion failed, but markdown is available")
        
        print("✓ Martensite review completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())