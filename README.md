![Martensite Banner](assets/banner.png)

# Martensite

*Adversarial hardening for modern grantsmanship*

A multi-LLM application review system for strengthening academic grant applications through adversarial critique from diverse AI perspectives.

## Overview

**Martensite** orchestrates multiple large language models to act as adversarial reviewers, identifying weaknesses, inconsistencies, and areas for improvement in academic documents. By aggregating feedback from diverse AI perspectives, it provides comprehensive, actionable insights to strengthen applications before submission.

The name "Martensite" derives from the hard, strong crystalline structure formed during rapid quenching of steel - analogous to how this system strengthens applications through rigorous AI-driven critique.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run a review with inline prompt
martensite -a application.pdf -P "Review this carefully"

# Run with call documentation
martensite -a application.pdf -c CallTexts/ -P "Review against these criteria"

# Run with custom output location
martensite -a CV.pdf -P "Focus on research impact" -o cv_review.pdf
```

## Installation

### 1. Add Martensite to Your PATH

Add the Martensite directory to your shell's PATH:

```bash
# Add this line to your ~/.bashrc
export PATH="$PATH:/path/to/Martensite"
```

Then reload your shell configuration:
```bash
source ~/.bashrc
```

### 2. Alternative: Create a Symbolic Link

Or create a symbolic link in a directory already in your PATH:

```bash
ln -s /path/to/Martensite/martensite.sh ~/.local/bin/martensite
```

### 3. Install Dependencies

**Python dependencies:**
```bash
pip install -r requirements.txt
```

**System dependencies (Linux - Arch/Manjaro):**
```bash
sudo pacman -S pandoc texlive-core texlive-latexextra poppler
```

**System dependencies (Linux - Debian/Ubuntu):**
```bash
sudo apt install pandoc texlive-xetex texlive-latex-extra poppler-utils
```

**System dependencies (macOS):**
```bash
brew install pandoc poppler
brew install --cask mactex  # Or: brew install texlive
```

## Platform Support

✅ **Linux**: Fully supported (tested on Arch, Ubuntu, Debian)
✅ **macOS**: Fully supported (10.15+ recommended)

## Features

✅ **Multi-LLM Reviews**: OpenAI (GPT-4o, GPT-5, o4-mini), Anthropic (Claude), Google (Gemini)
✅ **Secure API Key Discovery**: Environment → keyring (Keychain/Secret Service) → config file
✅ **Token Usage Tracking**: Monitor input/output tokens per model
✅ **PDF + Markdown Output**: Both formats always generated
✅ **Call Documentation Support**: Include grant call criteria in reviews
✅ **Modular Python Package**: Clean architecture for easy extension
✅ **Cross-Platform**: Works seamlessly on Linux and macOS

## Usage

### Basic Review

```bash
martensite -a StatementOfPurpose.pdf -P "Review this research proposal"
```

### With Call Documentation

```bash
martensite -a proposal.pdf -c CallTexts/ -P "Evaluate against call criteria"
```

### Custom Output Path

```bash
martensite -a CV.pdf -P "Focus on career progression" -o reviews/cv_review.pdf
```

### Dry Run (Testing)

```bash
martensite -a proposal.pdf -P "Test prompt" -o test.pdf -d
```

## Command-Line Options

- `-a, --application`: Path to PDF application file (required)
- `-p, --prompt`: Path to text file containing review prompt
- `-P, --prompt-string`: Review prompt as inline string
- `-c, --call-docs`: Path to directory or PDF with call documentation
- `-o, --output`: Output path for generated PDF (default: martensite.pdf)
- `-d, --dry-run`: Skip API calls and generate dummy reviews for testing
- `-h, --help`: Show help message

## Project Structure

```
Martensite/
├── martensite/              # Core Python package
│   ├── __init__.py
│   ├── key_discovery.py     # Secure API key management
│   ├── application_reviewer.py  # Multi-LLM orchestration
│   ├── martensite_handler.py    # CLI backend handler
│   └── extract_pdf_text.py # PDF text extraction utilities
├── tests/                   # Test suite
│   ├── LaCaixaIASTRO/      # Test application files
│   └── *.sh                # Test scripts
├── martensite.sh           # Main CLI entry point
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Architecture

```
PDF Application + Call Docs
        ↓
Multi-LLM Dispatcher
   ↓  ↓  ↓  ↓  ↓
GPT-4o Claude Gemini o4-mini GPT-5
   ↓  ↓  ↓  ↓  ↓
Response Aggregator
        ↓
Markdown + PDF Output
```

## Core Components

### 1. Multi-LLM Orchestration (`application_reviewer.py`)
- **Parallel API calls** across OpenAI, Anthropic, and Google models
- **Configurable model selection** for diverse perspectives
- **PDF processing** with text extraction
- **Async execution** for maximum performance
- **Comprehensive error handling** and retry logic

### 2. Secure Key Management (`key_discovery.py`)
- **Environment variables** checked first
- **System keyring** integration (keyctl/Keychain)
- **XDG config directory** fallback (~/.config/martensite/)
- **Multi-provider support** for OpenAI, Anthropic, Google

### 3. CLI Handler (`martensite_handler.py`)
- **Argument parsing** and validation
- **PDF text extraction** using PyPDF2
- **Markdown generation** with structured output
- **PDF conversion** via pandoc/XeLaTeX
- **Token usage reporting**

## Supported Models

### OpenAI
- **GPT-5**: Most advanced reasoning model
- **GPT-4o**: Flagship multimodal model
- **o4-mini-2025-04-16**: Cost-effective reasoning model

### Anthropic
- **claude-sonnet-4-5-20250929**: Latest flagship model

### Google
- **gemini-2.5-pro**: Advanced multimodal model

## Use Cases

1. **Grant Application Review**: Systematic evaluation of research proposals
2. **Academic Paper Critique**: Pre-submission peer review simulation
3. **Fellowship Applications**: Comprehensive application strengthening
4. **Research Statement Analysis**: Methodology and feasibility assessment
5. **Teaching Statement Review**: Pedagogical approach evaluation

## Output Format

Martensite generates both Markdown and PDF outputs:

### Markdown Structure
- **Header**: Banner, timestamp, input files
- **API Call Log**: Table with model, status, duration, token counts
- **Prompt**: The review prompt provided
- **Individual Reviews**: One per model, with timestamp
- **Consensus Summary**: (Disabled by default)

### PDF Features
- **Custom headers**: Section names in left header, timestamp in right
- **Page numbering**: "Page X of Y" in footer
- **Clean typography**: Via XeLaTeX with DejaVu Sans font
- **Professional formatting**: Proper spacing and structure

## Troubleshooting

**"Command not found"**: Check your PATH or use full path to martensite.sh

**PDF extraction fails**: Install PyPDF2 (`pip install PyPDF2`)

**PDF conversion fails**:
- Linux: `sudo apt install pandoc texlive-xetex` or `sudo pacman -S pandoc texlive-core`
- macOS: `brew install pandoc` and `brew install --cask mactex`

**Unicode errors**: Ensure XeLaTeX is installed with Unicode font support

**API key not found**:
- Set environment variables: `export OPENAI_API_KEY=sk-...`
- Or use OS keyring: `keyring set llm/openai default` (then paste key)
- Or create config file:
  - Linux: `~/.config/llm-keys/config.toml`
  - macOS: `~/Library/Application Support/llm-keys/config.toml`

**Model fails**: Check API call log in output for specific error messages

**macOS: "Operation not permitted"**: Grant Terminal full disk access in System Preferences → Security & Privacy

## Scientific Validation

The multi-LLM approach provides:
- **Reduced bias** through model diversity
- **Increased coverage** of potential issues
- **Statistical robustness** through repeated sampling
- **Diverse perspectives** from different model architectures

## Integration Points

- **LaTeX Workflow**: Direct PDF processing from compiled documents
- **Version Control**: Track improvements across document iterations
- **Academic Timelines**: Integrate with application deadline management
- **Zathura/PDF Viewers**: Auto-refresh for iterative development

## License

See LICENSE file for details.

---

*Martensite represents a paradigm shift in academic application preparation - from subjective self-assessment to objective, multi-perspective AI evaluation.*
