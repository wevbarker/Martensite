#!/usr/bin/env python3
"""
Multi-LLM Application Review System (EXPERIMENTAL - NOT IN USE)

This module provides an alternative implementation with native PDF support
for Anthropic and Google models. It is NOT currently wired into the CLI.

Current status: Reference implementation
Active implementation: martensite_handler.py (text extraction approach)

Future work: Integrate this class-based approach to leverage native PDF APIs
where available (Claude, Gemini) while maintaining text extraction fallback
for models without PDF support (OpenAI GPT).
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any
import base64
import aiohttp
import openai
from anthropic import Anthropic
import google.generativeai as genai

@dataclass
class ReviewConfig:
    """Configuration for the review system"""
    pdf_path: str
    base_prompt: str
    section_focus: str
    models: List[str]
    calls_per_model: int
    api_keys: Dict[str, str]
    
@dataclass
class ReviewResult:
    """Single review result"""
    model: str
    call_id: int
    response: str
    timestamp: str

class ApplicationReviewer:
    """Multi-LLM application review orchestrator"""
    
    def __init__(self, config: ReviewConfig):
        self.config = config
        self.results: List[ReviewResult] = []
        self.setup_clients()
        
    def setup_clients(self):
        """Initialize API clients"""
        if 'openai' in self.config.api_keys:
            openai.api_key = self.config.api_keys['openai']
        if 'anthropic' in self.config.api_keys:
            self.anthropic = Anthropic(api_key=self.config.api_keys['anthropic'])
        if 'google' in self.config.api_keys:
            genai.configure(api_key=self.config.api_keys['google'])
    
    def encode_pdf(self) -> str:
        """Encode PDF to base64 for API calls"""
        with open(self.config.pdf_path, 'rb') as f:
            return base64.b64encode(f.read()).decode()
    
    def build_prompt(self) -> str:
        """Build the complete prompt"""
        return f"""I am refereeing for a competitive research fellowship. Here is an application I have been sent. 

Please judge the section on: {self.config.section_focus}

{self.config.base_prompt}

Please be critical but fair, focusing on:
1. Scientific rigor and novelty
2. Feasibility and timeline realism  
3. Clarity of presentation
4. Potential impact
5. Any weaknesses or concerns

Provide specific, actionable feedback."""

    async def call_openai_model(self, model: str, call_id: int) -> ReviewResult:
        """Call OpenAI models (GPT-4, o1, etc.)"""
        try:
            if model.startswith('o1'):
                # o1 models don't support system messages
                response = await openai.ChatCompletion.acreate(
                    model=model,
                    messages=[{"role": "user", "content": self.build_prompt()}],
                    max_completion_tokens=4000
                )
            else:
                response = await openai.ChatCompletion.acreate(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are an expert academic referee reviewing grant applications."},
                        {"role": "user", "content": self.build_prompt()}
                    ],
                    max_tokens=2000,
                    temperature=0.7
                )

            return ReviewResult(
                model=model,
                call_id=call_id,
                response=response['choices'][0]['message']['content'],
                timestamp=datetime.now().isoformat()
            )
        except Exception as e:
            logging.error(f"OpenAI call failed for {model}: {e}")
            return None

    async def call_anthropic_model(self, model: str, call_id: int) -> ReviewResult:
        """Call Anthropic models (Claude)"""
        try:
            pdf_data = self.encode_pdf()
            
            response = await self.anthropic.messages.create(
                model=model,
                max_tokens=2000,
                temperature=0.7,
                messages=[{
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": self.build_prompt()},
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf", 
                                "data": pdf_data
                            }
                        }
                    ]
                }]
            )

            return ReviewResult(
                model=model,
                call_id=call_id,
                response=response.content[0].text,
                timestamp=datetime.now().isoformat()
            )
        except Exception as e:
            logging.error(f"Anthropic call failed for {model}: {e}")
            return None

    async def call_google_model(self, model: str, call_id: int) -> ReviewResult:
        """Call Google models (Gemini)"""
        try:
            # Upload PDF
            pdf_file = genai.upload_file(self.config.pdf_path)
            
            model_client = genai.GenerativeModel(model)
            response = await model_client.generate_content_async([
                pdf_file,
                self.build_prompt()
            ])

            return ReviewResult(
                model=model,
                call_id=call_id,
                response=response.text,
                timestamp=datetime.now().isoformat()
            )
        except Exception as e:
            logging.error(f"Google call failed for {model}: {e}")
            return None


    async def run_reviews(self) -> List[ReviewResult]:
        """Run all review calls"""
        tasks = []
        
        for model in self.config.models:
            for call_id in range(self.config.calls_per_model):
                if model.startswith('gpt-') or model.startswith('o1-'):
                    task = self.call_openai_model(model, call_id)
                elif model.startswith('claude-'):
                    task = self.call_anthropic_model(model, call_id)
                elif model.startswith('gemini-'):
                    task = self.call_google_model(model, call_id)
                else:
                    continue
                
                tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        self.results = [r for r in results if isinstance(r, ReviewResult)]
        return self.results

    async def summarize_criticisms(self) -> str:
        """Generate meta-summary of shared criticisms"""
        if not self.results:
            return "No results to summarize"
        
        # Compile all responses
        all_responses = "\n\n=== REVIEW SEPARATOR ===\n\n".join([
            f"Model: {r.model} (Call {r.call_id})\n{r.response}" 
            for r in self.results
        ])
        
        summary_prompt = f"""Here are multiple independent reviews of the same grant application section:

{all_responses}

Please summarize the main criticisms that are shared across multiple reviews. Focus on:
1. Issues mentioned by 2+ different models
2. Consistent patterns of concern
3. Actionable improvements suggested by multiple reviewers

Ignore one-off comments and focus on consensus criticisms."""

        # Use most reliable model for summary
        try:
            response = await openai.ChatCompletion.acreate(
                model="gpt-4o",
                messages=[{"role": "user", "content": summary_prompt}],
                max_tokens=1000,
                temperature=0.3
            )
            return response['choices'][0]['message']['content']
        except:
            return "Failed to generate summary"

    def save_results(self, output_path: str):
        """Save results to JSON file"""
        data = {
            'config': {
                'section_focus': self.config.section_focus,
                'models': self.config.models,
                'calls_per_model': self.config.calls_per_model
            },
            'results': [
                {
                    'model': r.model,
                    'call_id': r.call_id,
                    'response': r.response,
                    'timestamp': r.timestamp
                } for r in self.results
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

# Example usage
async def main():
    config = ReviewConfig(
        pdf_path="StatementOfPurpose.pdf",
        base_prompt="Focus on scientific rigor and innovation.",
        section_focus="Research Objectives and Methodology",
        models=[
            "gpt-4o",
            "gpt-4o-mini", 
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "gemini-1.5-pro"
        ],
        calls_per_model=3,
        api_keys={
            'openai': 'your-openai-key',
            'anthropic': 'your-anthropic-key', 
            'google': 'your-google-key'
        }
    )
    
    reviewer = ApplicationReviewer(config)
    results = await reviewer.run_reviews()
    summary = await reviewer.summarize_criticisms()
    
    print(f"Completed {len(results)} reviews")
    print(f"\nSummary of shared criticisms:\n{summary}")
    
    reviewer.save_results("review_results.json")

if __name__ == "__main__":
    asyncio.run(main())