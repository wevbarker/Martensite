# Martensite

*Adversarial hardening for modern grantsmanship*

A multi-LLM application review system for strengthening academic grant applications through adversarial critique from diverse AI perspectives.

## Project Structure

```
Martensite/
├── martensite/              # Core Python package
│   ├── __init__.py
│   ├── key_discovery.py     # Secure API key management
│   ├── application_reviewer.py  # Multi-LLM orchestration
│   └── martensite_handler.py    # CLI backend handler
├── tests/                   # Test suite
│   ├── LaCaixaIASTRO/      # Test application files
│   └── *.py                # Test scripts
├── scripts/                 # Utility scripts
│   ├── extract_pdf_text.py
│   ├── cost_calculator.py
│   └── calculate_actual_costs.py
├── docs/                    # Documentation
│   ├── README.md           # Full documentation
│   ├── INSTALL.md          # Installation guide
│   └── PDF_STRATEGY_ANALYSIS.md
├── config/                  # Configuration files
├── martensite.sh           # Main CLI entry point
└── requirements.txt        # Python dependencies
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run a review
martensite -a application.pdf -p prompt.txt

# Or with inline prompt
martensite -a application.pdf -P "Review this carefully"
```

## Documentation

See [`docs/README.md`](docs/README.md) for full documentation and [`docs/INSTALL.md`](docs/INSTALL.md) for installation instructions.

## Features

✅ Multi-LLM reviews (OpenAI, Anthropic, Google/Gemini)
✅ Secure API key discovery (environment → keyring → XDG config)
✅ Cost tracking and optimization
✅ PDF + Markdown output
✅ Modular Python package architecture

## License

See LICENSE file for details.
