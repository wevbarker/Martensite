# Martensite Installation Instructions

*Adversarial hardening for modern grantsmanship*

## What You Need to Do

### 1. Add Martensite to Your PATH

Add the Martensite directory to your shell's PATH so you can run `martensite` from anywhere:

```bash
# Add this line to your ~/.bashrc (it may already be there from earlier)
export PATH="$PATH:/home/barker/Documents/Martensite"
```

Then reload your shell configuration:
```bash
source ~/.bashrc
```

### 2. Create a Symbolic Link (Alternative Method)

Or create a symbolic link in a directory that's already in your PATH:

```bash
ln -s /home/barker/Documents/Martensite/martensite.sh /home/barker/.local/bin/martensite
```

### 3. Install Dependencies (Optional but Recommended)

For PDF text extraction:
```bash
pip install PyPDF2
```

For PDF conversion (to fix the Unicode issue):
```bash
# Install pandoc with better Unicode support
sudo pacman -S pandoc texlive-core texlive-latexextra
```

## Usage

Once installed, you can run `martensite` from any application directory:

```bash
# Basic usage
martensite -a StatementOfPurpose.pdf -p review_prompt.txt

# With custom output
martensite -a CV.pdf -p cv_questions.txt -o detailed_cv_review.pdf

# Help
martensite -h
```

## Features

✅ **Multi-LLM Reviews**: Uses GPT-4o-mini and GPT-4o for balanced cost/quality  
✅ **Cost Tracking**: Shows exact cost for each review  
✅ **Append Mode**: Multiple reviews accumulate in the same file  
✅ **Timestamped**: Each review session has a clear timestamp  
✅ **Markdown + PDF**: Outputs both formats  
✅ **Token Capping**: 500 tokens max per response for cost control  

## Cost Estimates

- **Typical Document Review**: $0.10-$0.15
- **Monthly Budget**: $100 = ~700 reviews
- **Models Used**: GPT-4o-mini (cost-effective) + GPT-4o (higher quality)

## File Structure

```
Martensite/
├── martensite.sh          # Main command script
├── martensite_handler.py  # Python backend
├── example_prompt.txt     # Sample prompt
├── INSTALL.md            # This file
├── extract_pdf_text.py   # PDF extraction utility
├── cost_calculator.py    # Cost analysis
└── application_reviewer.py # Multi-LLM orchestration
```

## Example Workflow

1. Navigate to your application directory:
   ```bash
   cd /path/to/your/application
   ```

2. Create a custom prompt file:
   ```bash
   echo "Please review this CV focusing on research impact and career progression." > cv_prompt.txt
   ```

3. Run martensite:
   ```bash
   martensite -a CV.pdf -p cv_prompt.txt -o cv_review.pdf
   ```

4. View results:
   ```bash
   # Markdown (always available)
   cat cv_review.md
   
   # PDF (if pandoc conversion succeeded)
   evince cv_review.pdf
   ```

## Troubleshooting

**"Command not found"**: Check your PATH or use the full path to martensite.sh

**PDF extraction fails**: Install PyPDF2 or ensure pdftotext is available

**PDF conversion fails**: Install pandoc and LaTeX packages

**Unicode errors**: Use a Unicode-aware LaTeX distribution or stick with markdown output