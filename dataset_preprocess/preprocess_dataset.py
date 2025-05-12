import re
from typing import List, Dict
import math

def clean_formula(text: str) -> str:
    """Clean mathematical formulas in the text"""
    # Replace complex LaTeX/Wikipedia style formulas with simpler versions
    formula_replacements = {
        # Basic operations
        r'\\dfrac': '/',
        r'\{': '',
        r'\}': '',
        r'\\rightarrow': '->',
        r'\\infty': 'infinity',
        r'\\left': '',
        r'\\right': '',
        r'\\displaystyle': '',

        # Common mathematical symbols
        r'\\sum': 'sum',
        r'\\prod': 'product',
        r'\\sqrt': 'sqrt',
        r'\\pi': 'pi',
        r'\\theta': 'theta',
        r'\\alpha': 'alpha',
        r'\\beta': 'beta',
        r'\\sigma': 'sigma',
        r'\\mu': 'mu',
        r'\\Delta': 'Delta',

        # Special functions
        r'\\log': 'log',
        r'\\ln': 'ln',
        r'\\exp': 'exp',
        r'\\sin': 'sin',
        r'\\cos': 'cos',
        r'\\tan': 'tan'
    }

    # Apply all replacements
    for latex, simple in formula_replacements.items():
        text = text.replace(latex, simple)

    # Clean up remaining LaTeX-style formatting
    text = re.sub(r'[\$]{1,2}.*?[\$]{1,2}', '', text)  # Remove inline LaTeX
    text = re.sub(r'\\[a-zA-Z]+', '', text)  # Remove remaining LaTeX commands

    return text

def clean_references(text: str) -> str:
    """Remove reference markers and clean up reference-related formatting"""
    # Remove reference citations [1], [2], etc.
    text = re.sub(r'\[\d+\]', '', text)
    # Remove reference links
    text = re.sub(r'\[\[.*?\]\]', '', text)
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    return text

def clean_section_headers(text: str) -> str:
    """Clean up section headers while preserving important content"""
    # Remove == style headers except for Topic
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        if not line.strip().startswith('==') or 'TOPIC:' in line:
            cleaned_lines.append(line)
    return '\n'.join(cleaned_lines)

def preprocess_finance_text(text: str) -> str:
    """Main preprocessing function for financial text data"""

    # Extract only the main content (remove headers except TOPIC)
    main_content = ''
    sections = text.split('=' * 80)
    for section in sections:
        if section.strip():
            # Keep only TOPIC line and main content
            lines = section.strip().split('\n')
            filtered_lines = []
            for line in lines:
                if line.startswith('TOPIC:') or (not line.startswith('SOURCE:') and '=' * 20 not in line):
                    filtered_lines.append(line)
            main_content += '\n'.join(filtered_lines) + '\n\n'

    # Clean up the text
    cleaned_text = main_content.strip()
    cleaned_text = clean_formula(cleaned_text)
    cleaned_text = clean_references(cleaned_text)
    cleaned_text = clean_section_headers(cleaned_text)

    # Additional cleaning steps
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Replace multiple spaces with single space
    cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)  # Normalize multiple newlines
    cleaned_text = cleaned_text.strip()

    # Split into paragraphs and append <|endoftext|> token to each paragraph except TOPIC lines
    paragraphs = cleaned_text.split('\n')
    final_paragraphs = []
    for paragraph in paragraphs:
        if paragraph.startswith('TOPIC:'):
            final_paragraphs.append(paragraph)
        else:
            final_paragraphs.append(paragraph + ' <|endoftext|>')

    # Join paragraphs back
    final_text = '\n'.join(final_paragraphs)

    return final_text

def prepare_for_gpt_training(input_file: str, output_file: str) -> None:
    """Prepare the entire dataset for GPT training"""
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()

        # Split into individual documents
        documents = text.split('=' * 80)

        # Process each document
        processed_documents = []
        for doc in documents:
            if doc.strip():
                processed_text = preprocess_finance_text(doc)
                if processed_text.strip():
                    processed_documents.append(processed_text)

        # Combine processed documents
        final_text = '\n\n' + '=' * 40 + '\n\n'.join(processed_documents)

        # Save processed text
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(final_text)

        print(f"Successfully processed data and saved to {output_file}")
        print(f"Added {len(processed_documents)} processed documents")

    except Exception as e:
        print(f"Error processing file: {e}")

# Example usage
if __name__ == "__main__":
    prepare_for_gpt_training('enhanced_finance_dataset.txt', 'processed_finance_dataset.txt')
