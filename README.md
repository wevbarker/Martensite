[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![Martensite Banner](martensite/banner.png)

# Martensite

*Adversarial hardening for modern grantsmanship*

## Logos (this part was written by a human)

Grantsmanship - the process of securing grants - is changing due to AI. This is an unprecedented, irreversible and unavoidable process, occurring at a rate which exceeds the adaptive capacity of all institutions and most individuals. From the accelerationist perspective, it is necessary to _lean in_ to emerging technologies and ensure that they serve to maximise scientific productivity. This is, moreover, a duty incumbent on all researchers, owing to the importance of public trust in the academic enterprise.

Within Europe, applications for prestigious funding opportunities have risen sharply since 2022. Applicants are using LLMs to craft and refine their proposals, and referees are using LLMs to evaluate them. A minority of both applicants and referees deny this, including a dwindling subset who are telling the truth. Considering only the tools that are available at the time of writing, academia will naturally saturate in a configuration where LLM-assisted grantsmanship is not only ubiquitous, but also acceptable, and even expected. It will be argued elsewhere that this will have several benefits, including but not limited to:
- The liberation of time which can be spent on research, constituting a better use of public money.
- Mitigation of disadvantages which are utterly irrelevant to caliber of the researcher, such as being a non-native English speaker.
- Improved signal-to-noise ratio in gauging the excellence of ideas, due to all proposals having - and being expected to have - flawless presentation.
- A general diminution of grantsmanship itself as a proxy metric, and a return to results-based evaluation: "_Stop promising to do things. What did you publish with your last grant, and who cited you?_"

*Martensite* is intended to bring this saturation point forwards in time. The source code should be cloned, forked, or used a prompt for those wishing to roll their own system. The name derives from the hard, strong crystalline phase of steel formed by rapid quenching.

```bash
martensite -a application.pdf -c call_dir -P "I have been tasked with reviewing this research proposal, which is slightly outside of my area of expertise. Can you carefully examine the call text, and read the proposal, and draft a review for me?" -o report.pdf
```
Takes your research proposal `application.pdf` and the public call texts downloaded to `call_dir`, and generates multiple referee reports in `report.pdf`. The prompt is sent to the most advanced flagship models from _OpenAI_, _Anthropic_, and _Google_. You will need to have funded API keys for at least one provider.

## Quick Start

For confident super-users. If you're less comfortable in the terminal, proceed step-by-step through [Installation](#installation) below.

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

### 1. PATH Configuration

The PATH tells your terminal where to find commands. This step lets you type `martensite` from any directory instead of typing the full path to the program.

Add *Martensite* to your PATH using either method:

**Option A: Direct export**

This adds *Martensite* to your PATH permanently.

```bash
# Add this line to your ~/.bashrc or ~/.zshrc
export PATH="$PATH:/path/to/Martensite"

# Reload your shell configuration
source ~/.bashrc  # or source ~/.zshrc
```

**Option B: Symbolic link**

This creates a shortcut in a directory that's already in your PATH.

```bash
ln -s /path/to/Martensite/martensite.sh ~/.local/bin/martensite
```

### 2. Dependencies

*Martensite* is written in Python and generates referee reports as PDF files. You'll need to install some additional software packages.

**Python dependencies:**

These are Python libraries that *Martensite* uses to communicate with AI providers and process documents.

```bash
pip install -r requirements.txt
```

**Report dependencies:**

PDF generation requires pandoc (document converter), LaTeX (typesetting system), and PDF utilities.

***Linux* (*Arch*/Manjaro):**
```bash
sudo pacman -S pandoc texlive-core texlive-latexextra poppler
```

***Linux* (*Debian*/*Ubuntu*):**
```bash
sudo apt install pandoc texlive-xetex texlive-latex-extra poppler-utils
```

***macOS*:**
```bash
brew install pandoc poppler
brew install --cask mactex  # Or: brew install texlive
```

### 3. API Keys

*Martensite* supports three LLM providers. You need at least one API key to run reviews.

**What are API keys?**

If you've used *ChatGPT* through a web browser, you've been using *OpenAI*'s consumer interface. API keys let *Martensite* access the same AI models programmatically, directly from your computer. Unlike the web interface with a monthly subscription, API access is pay-as-you-go: you add funds to your account and pay only for what you use (typically a few cents per review). You'll need to create a developer account with at least one provider (*OpenAI*, *Anthropic*, or *Google*) and add payment information before generating API keys.

**How *Martensite* finds your keys:**

*Martensite* checks three locations in order: environment variables → OS keyring → config file. Choose whichever method is most convenient for you.

**Option A: Environment variables (recommended)**
```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."
```

**Option B: OS keyring**
```bash
# Uses system Keychain (macOS) or Secret Service (Linux)
keyring set llm/openai default    # paste key when prompted
keyring set llm/anthropic default
keyring set llm/google default
```

**Option C: Config file**

Create `~/.config/llm-keys/config.toml` (*Linux*) or `~/Library/Application Support/llm-keys/config.toml` (*macOS*):

```toml
[openai]
api_key = "sk-..."

[anthropic]
api_key = "sk-ant-..."

[google]
api_key = "..."
```

## Platform Support

- ***Linux***: Fully supported (tested on *Arch*, *Ubuntu*, *Debian*)
- ***macOS***: Fully supported (10.15+ recommended)

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

## Core Components

### 1. Multi-LLM Orchestration (`application_reviewer.py`)
- **Parallel API calls** across *OpenAI*, *Anthropic*, and *Google* models
- **Configurable model selection** for diverse perspectives
- **PDF processing** with text extraction
- **Async execution** for maximum performance
- **Comprehensive error handling** and retry logic

### 2. Secure Key Management (`key_discovery.py`)
- **Environment variables** checked first
- **System keyring** integration (keyctl/Keychain)
- **XDG config directory** fallback (~/.config/martensite/)
- **Multi-provider support** for *OpenAI*, *Anthropic*, *Google*

### 3. CLI Handler (`martensite_handler.py`)
- **Argument parsing** and validation
- **PDF text extraction** using PyPDF2
- **Markdown generation** with structured output
- **PDF conversion** via pandoc/XeLaTeX
- **Token usage reporting**

## Supported Models

### *OpenAI*
- **GPT-5**: Most advanced reasoning model
- **GPT-4o**: Flagship multimodal model
- **o4-mini-2025-04-16**: Cost-effective reasoning model

### *Anthropic*
- **claude-sonnet-4-5-20250929**: Latest flagship model

### *Google*
- **gemini-2.5-pro**: Advanced multimodal model

## Troubleshooting

**"Command not found"**: Check your PATH or use full path to martensite.sh

**PDF extraction fails**: Install PyPDF2 (`pip install PyPDF2`)

**PDF conversion fails**:
- *Linux*: `sudo apt install pandoc texlive-xetex` or `sudo pacman -S pandoc texlive-core`
- *macOS*: `brew install pandoc` and `brew install --cask mactex`

**Unicode errors**: Ensure XeLaTeX is installed with Unicode font support

**API key not found**:
- Set environment variables: `export OPENAI_API_KEY=sk-...`
- Or use OS keyring: `keyring set llm/openai default` (then paste key)
- Or create config file:
  - *Linux*: `~/.config/llm-keys/config.toml`
  - *macOS*: `~/Library/Application Support/llm-keys/config.toml`

**Model fails**: Check API call log in output for specific error messages

***macOS*: "Operation not permitted"**: Grant Terminal full disk access in System Preferences → Security & Privacy

## License

See LICENSE file for details.

## Acknowledgements

I am indebted to Will Handley for (i) calling out zero-sum famine mentality on my part, and encouraging me to make this repository public, and (ii) funding the _OpenAI_ inference used during development. I am also grateful to Ilona Gottwaldová for bringing the big picture of LLM-assisted grantsmanship to my attention.
