# Martensite

*Adversarial hardening for modern grantsmanship*

**Martensite** is a comprehensive system for "hardening" academic grant applications and research documents through adversarial multi-LLM review. The name "Martensite" derives from the hard, strong crystalline structure formed during rapid quenching of steel - analogous to how this system strengthens applications through rigorous AI-driven critique.

## Overview

Martensite orchestrates multiple large language models to act as adversarial reviewers, identifying weaknesses, inconsistencies, and areas for improvement in academic documents. By aggregating feedback from diverse AI perspectives, it provides comprehensive, actionable insights to strengthen applications before submission.

## Core Components

### 1. Multi-LLM Orchestration (`application_reviewer.py`)
- **Parallel API calls** across OpenAI, Anthropic, and Google models
- **Configurable model selection** with cost optimization
- **PDF processing** with direct upload to compatible models
- **Async execution** for maximum performance
- **Comprehensive error handling** and retry logic

### 2. Cost Analysis (`cost_calculator.py`)
- **Real-time pricing** based on current API rates
- **Three-tier system**: Minimal ($0.08), Balanced ($0.51), Premium ($3.11) per section
- **Budget planning** with monthly review capacity estimates
- **Token estimation** for accurate cost prediction

### 3. Consensus Analysis
- **Criticism aggregation** across multiple model responses
- **Pattern recognition** for issues mentioned by 2+ models
- **Actionable feedback prioritization** based on consensus
- **Meta-analysis** using advanced summarization techniques

## Key Features

### Multi-Provider Support
- **OpenAI**: GPT-4o, GPT-4o-mini, o1-preview, o1-mini
- **Anthropic**: Claude-3.5-Sonnet, Claude-3.5-Haiku, Claude-3-Opus
- **Google**: Gemini-1.5-Pro, Gemini-1.5-Flash

### Cost Optimization
```
Scenario 1: Minimal Setup    - $0.08 per section review
Scenario 2: Balanced Setup   - $0.51 per section review  
Scenario 3: Premium Setup    - $3.11 per section review
```

### Scalability
- **Configurable parallelism** with multiple calls per model
- **Statistical significance** through repeated evaluations
- **JSON export** for result archival and analysis
- **Modular architecture** for easy extension

## Use Cases

1. **Grant Application Review**: Systematic evaluation of research proposals
2. **Academic Paper Critique**: Pre-submission peer review simulation
3. **Fellowship Applications**: Comprehensive application strengthening
4. **Research Statement Analysis**: Methodology and feasibility assessment
5. **Teaching Statement Review**: Pedagogical approach evaluation

## Architecture

```
PDF Application Input
        ↓
Multi-LLM Dispatcher
   ↓  ↓  ↓  ↓  ↓
GPT-4o Claude Gemini [...]
   ↓  ↓  ↓  ↓  ↓
Response Aggregator
        ↓
Consensus Analyzer
        ↓
Actionable Report
```

## Integration Points

- **LaTeX Workflow**: Direct PDF processing from compiled documents
- **Spell-checking**: Complement existing linguistic analysis tools
- **Version Control**: Track improvements across document iterations
- **Academic Timelines**: Integrate with application deadline management

## Scientific Validation

The multi-LLM approach provides:
- **Reduced bias** through model diversity
- **Increased coverage** of potential issues
- **Statistical robustness** through repeated sampling
- **Consensus-based confidence** in feedback prioritization

## Directory Structure

```
Martensite/
├── README.md                    # This file
├── martensite.sh               # Main CLI interface
├── martensite_handler.py       # Python orchestration handler
├── application_reviewer.py     # Core multi-LLM orchestration
├── cost_calculator.py          # Pricing analysis and optimization
├── config/                     # Configuration templates
├── tests/                      # Test cases and examples
└── scripts/                    # Utility and automation scripts
```

## Next Steps

1. **Configuration System**: Template-based reviewer prompts
2. **Report Generation**: Automated LaTeX report compilation  
3. **Iterative Hardening**: Closed-loop improvement cycles
4. **Integration Testing**: Real-world application validation
5. **Performance Benchmarking**: Model comparison and optimization

---

*Martensite represents a paradigm shift in academic application preparation - from subjective self-assessment to objective, multi-perspective AI evaluation.*