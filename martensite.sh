#!/bin/bash
#
# MARTENSITE - Adversarial hardening for modern grantsmanship
# Usage: martensite -a PathToApplication.pdf {-p PathToPrompt.txt | -P "prompt string"} [-c PathToCallDocs] [-o PathToOutput.pdf]
#

set -e  # Exit on error

# Default values
OUTPUT_PATH="./martensite.pdf"
CALL_DOCS_PATH=""
PROMPT_FILE=""
PROMPT_STRING=""
DRY_RUN=""
# Get the real directory of the script (resolve symlinks) - portable version
# Works on both Linux and macOS
SCRIPT_PATH="$(python3 -c "import os, sys; print(os.path.realpath(sys.argv[1]))" "${BASH_SOURCE[0]}")"
MARTENSITE_DIR="$(dirname "$SCRIPT_PATH")"
PYTHON_SCRIPT="${MARTENSITE_DIR}/martensite/martensite_handler.py"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to display usage
usage() {
    echo "Usage: martensite -a PathToApplication.pdf {-p PathToPrompt.txt | -P \"prompt string\"} [-c PathToCallDocs] [-o PathToOutput.pdf]"
    echo ""
    echo "Options:"
    echo "  -a    Path to the application PDF to review"
    echo "  -p    Path to the prompt text file containing review questions"
    echo "  -P    Prompt string directly (alternative to -p)"
    echo "  -c    Path to call documentation (PDF/HTML files or directory containing them)"
    echo "  -o    Path to output PDF (default: ./martensite.pdf)"
    echo "  -h    Display this help message"
    echo ""
    echo "Example:"
    echo "  martensite -a StatementOfPurpose.pdf -p review_prompt.txt"
    echo "  martensite -a CV.pdf -P \"Review this CV carefully\" -o cv_review.pdf"
    exit 1
}

# Parse command line arguments
while getopts "a:p:P:c:o:hd" opt; do
    case $opt in
        a)
            APPLICATION_PDF="$OPTARG"
            ;;
        p)
            PROMPT_FILE="$OPTARG"
            ;;
        P)
            PROMPT_STRING="$OPTARG"
            ;;
        c)
            CALL_DOCS_PATH="$OPTARG"
            ;;
        o)
            OUTPUT_PATH="$OPTARG"
            ;;
        d)
            DRY_RUN="--dry-run"
            ;;
        h)
            usage
            ;;
        \?)
            echo -e "${RED}Invalid option: -$OPTARG${NC}" >&2
            usage
            ;;
    esac
done

# Validate required arguments
if [ -z "$APPLICATION_PDF" ]; then
    echo -e "${RED}Error: -a argument is required${NC}"
    usage
fi

if [ -z "$PROMPT_FILE" ] && [ -z "$PROMPT_STRING" ]; then
    echo -e "${RED}Error: Either -p or -P argument is required${NC}"
    usage
fi

if [ -n "$PROMPT_FILE" ] && [ -n "$PROMPT_STRING" ]; then
    echo -e "${RED}Error: Cannot use both -p and -P arguments${NC}"
    usage
fi

# Validate file existence
if [ ! -f "$APPLICATION_PDF" ]; then
    echo -e "${RED}Error: Application PDF not found: $APPLICATION_PDF${NC}"
    exit 1
fi

# Handle prompt string by creating temporary file if needed
if [ -n "$PROMPT_STRING" ]; then
    TEMP_PROMPT_FILE=$(mktemp)
    echo "$PROMPT_STRING" > "$TEMP_PROMPT_FILE"
    PROMPT_FILE="$TEMP_PROMPT_FILE"
    trap "rm -f $TEMP_PROMPT_FILE" EXIT
fi

if [ ! -f "$PROMPT_FILE" ]; then
    echo -e "${RED}Error: Prompt file not found: $PROMPT_FILE${NC}"
    exit 1
fi

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo -e "${RED}Error: Martensite handler script not found: $PYTHON_SCRIPT${NC}"
    echo "Please ensure martensite_handler.py exists in the Martensite directory"
    exit 1
fi

# Ensure output directory exists
OUTPUT_DIR=$(dirname "$OUTPUT_PATH")
mkdir -p "$OUTPUT_DIR"

# Get absolute paths (portable - works on Linux and macOS)
APPLICATION_PDF=$(python3 -c "import os, sys; print(os.path.realpath(sys.argv[1]))" "$APPLICATION_PDF")
PROMPT_FILE=$(python3 -c "import os, sys; print(os.path.realpath(sys.argv[1]))" "$PROMPT_FILE")
OUTPUT_PATH=$(python3 -c "import os, sys; print(os.path.realpath(sys.argv[1]))" "$OUTPUT_PATH")

# Display configuration
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  MARTENSITE - Adversarial hardening for modern grantsmanship       ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Configuration:${NC}"
echo -e "  Application: $(basename "$APPLICATION_PDF")"
if [ -n "$PROMPT_STRING" ]; then
    echo -e "  Prompt:      [inline string]"
else
    echo -e "  Prompt:      $(basename "$PROMPT_FILE")"
fi
echo -e "  Output:      $OUTPUT_PATH"
echo ""

# Run the Python handler
echo -e "${YELLOW}Running multi-LLM review...${NC}"
# Build command with optional call docs parameter
CMD_ARGS=(
    --application "$APPLICATION_PDF"
    --prompt "$PROMPT_FILE"
    --output "$OUTPUT_PATH"
)

if [ -n "$CALL_DOCS_PATH" ]; then
    CMD_ARGS+=(--call-docs "$CALL_DOCS_PATH")
fi

if [ -n "$DRY_RUN" ]; then
    CMD_ARGS+=($DRY_RUN)
fi

python3 "$PYTHON_SCRIPT" "${CMD_ARGS[@]}"

# Check if successful
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Review completed successfully!${NC}"
    echo -e "${GREEN}✓ Output saved to: $OUTPUT_PATH${NC}"
    
    # Also show the markdown file location
    MD_PATH="${OUTPUT_PATH%.pdf}.md"
    if [ -f "$MD_PATH" ]; then
        echo -e "${GREEN}✓ Markdown saved to: $MD_PATH${NC}"
    fi
else
    echo -e "${RED}✗ Review failed. Check the error messages above.${NC}"
    exit 1
fi