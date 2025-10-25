#!/usr/bin/env python3
"""
Scrape P4F call texts and save as HTML and PDF
"""

import os
import re
import requests
from pathlib import Path
import subprocess
import time

# Target directory
OUTPUT_DIR = Path("tests/P4FSantoni/CallTexts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# URLs to scrape
URLS = {
    # PDFs (download directly)
    "pdfs": [
        "https://p4f.fzu.cz/wp-content/uploads/2025/06/P4F_guide_for_applicants_2nd_call_v2.pdf",
        "https://p4f.fzu.cz/wp-content/uploads/2025/06/P4F_Terms_and_conditions_2nd_call_v1.pdf",
        "https://p4f.fzu.cz/wp-content/uploads/2025/07/P4F_Guide_for_applicants_OnePager_Final.pdf",
        "https://p4f.fzu.cz/wp-content/uploads/2025/08/P4F_how_to_use_the_application_portal_2ndCall.pdf",
    ],
    # DOCX files (download and convert)
    "docx": [
        "https://p4f.fzu.cz/wp-content/uploads/2025/06/P4F_research_proposal_template.docx",
        "https://p4f.fzu.cz/wp-content/uploads/2025/06/Letter-of-Commitment-template.docx",
    ],
    # Web pages (scrape and save)
    "pages": [
        ("https://p4f.fzu.cz/", "P4F_homepage"),
        ("https://p4f.fzu.cz/for-candidates/", "P4F_for_candidates"),
        ("https://p4f.fzu.cz/our-supervisors/", "P4F_supervisors"),
        ("https://p4f.fzu.cz/news-p4f/", "P4F_news"),
        ("https://p4f.fzu.cz/news/2nd-call-open-application/", "P4F_2nd_call_announcement"),
        ("https://www.fzu.cz/en/research/projects/physics-future-p4f", "FZU_P4F_project_page"),
    ]
}

def download_file(url, output_path):
    """Download a file from URL"""
    print(f"Downloading {url} -> {output_path}")
    response = requests.get(url, allow_redirects=True)
    response.raise_for_status()
    with open(output_path, 'wb') as f:
        f.write(response.content)
    print(f"  Saved: {output_path}")

def download_pdfs():
    """Download PDF files directly"""
    for url in URLS["pdfs"]:
        filename = url.split('/')[-1].split('?')[0]  # Remove query params
        output_path = OUTPUT_DIR / filename
        try:
            download_file(url, output_path)
        except Exception as e:
            print(f"  ERROR downloading {url}: {e}")
        time.sleep(0.5)  # Be polite

def download_and_convert_docx():
    """Download DOCX files and convert to HTML and PDF"""
    for url in URLS["docx"]:
        filename = url.split('/')[-1].split('?')[0]
        docx_path = OUTPUT_DIR / filename

        try:
            # Download DOCX
            download_file(url, docx_path)

            # Convert to HTML using pandoc
            html_path = docx_path.with_suffix('.html')
            print(f"  Converting to HTML: {html_path}")
            subprocess.run([
                'pandoc', str(docx_path), '-o', str(html_path)
            ], check=True)

            # Convert to PDF using pandoc
            pdf_path = docx_path.with_suffix('.pdf')
            print(f"  Converting to PDF: {pdf_path}")
            subprocess.run([
                'pandoc', str(docx_path), '-o', str(pdf_path)
            ], check=True)

        except Exception as e:
            print(f"  ERROR processing {url}: {e}")

        time.sleep(0.5)

def scrape_pages():
    """Scrape web pages and save as HTML and PDF"""
    for url, name in URLS["pages"]:
        html_path = OUTPUT_DIR / f"{name}.html"
        pdf_path = OUTPUT_DIR / f"{name}.pdf"

        try:
            # Download page
            print(f"Scraping {url} -> {html_path}")
            response = requests.get(url)
            response.raise_for_status()

            # Save HTML
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
            print(f"  Saved HTML: {html_path}")

            # Convert to PDF using pandoc
            print(f"  Converting to PDF: {pdf_path}")
            subprocess.run([
                'pandoc', str(html_path), '-o', str(pdf_path),
                '--pdf-engine=xelatex'
            ], check=True, capture_output=True)
            print(f"  Saved PDF: {pdf_path}")

        except Exception as e:
            print(f"  ERROR processing {url}: {e}")

        time.sleep(1)  # Be polite

if __name__ == "__main__":
    print("Starting P4F call text scraping...")
    print(f"Output directory: {OUTPUT_DIR}\n")

    print("=" * 60)
    print("1. Downloading PDF files...")
    print("=" * 60)
    download_pdfs()

    print("\n" + "=" * 60)
    print("2. Downloading and converting DOCX files...")
    print("=" * 60)
    download_and_convert_docx()

    print("\n" + "=" * 60)
    print("3. Scraping web pages...")
    print("=" * 60)
    scrape_pages()

    print("\n" + "=" * 60)
    print("Done! Check", OUTPUT_DIR)
    print("=" * 60)
