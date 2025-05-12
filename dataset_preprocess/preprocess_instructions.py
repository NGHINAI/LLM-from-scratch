import json
import re


def standardize_notation(text):
    """
    Standardize mathematical notation while preserving LLM interpretability
    """
    # Keep common mathematical Unicode characters as they are
    preserved_chars = {
        # Superscripts for powers
        '\u00b2',  # ²
        '\u00b3',  # ³
        '\u2074',  # ⁴
        '\u2075',  # ⁵
        '\u2076',  # ⁶
        '\u2077',  # ⁷
        '\u2078',  # ⁸
        '\u2079',  # ⁹

        # Common mathematical symbols
        '\u221a',  # √ (square root)
        '\u03c0',  # π (pi)
        '\u2211',  # ∑ (summation)
        '\u222b',  # ∫ (integral)
        '\u00b1',  # ± (plus-minus)
    }

    # Dictionary for standardizing less common notations
    standardizations = {
        # Less common mathematical symbols that might need standardization
        '\u221b': '∛',  # cube root to standard symbol
        '\u221c': '∜',  # fourth root to standard symbol
        '\u2248': '≈',  # approximately equal
        '\u2260': '≠',  # not equal
        '\u2264': '≤',  # less than or equal
        '\u2265': '≥',  # greater than or equal

        # Greek letters (keep common ones, standardize uncommon ones)
        '\u03b1': 'α',  # alpha
        '\u03b2': 'β',  # beta
        '\u03b3': 'γ',  # gamma
        '\u03b4': 'δ',  # delta
        '\u03c3': 'σ',  # sigma
        '\u03bc': 'μ',  # mu
    }

    # First, handle special cases and formatting
    processed_text = text

    # Standardize spacing around mathematical operators
    processed_text = re.sub(r'(\d+)\s*×\s*(\d+)', r'\1 × \2', processed_text)
    processed_text = re.sub(r'(\d+)\s*\+\s*(\d+)', r'\1 + \2', processed_text)
    processed_text = re.sub(r'(\d+)\s*\-\s*(\d+)', r'\1 - \2', processed_text)
    processed_text = re.sub(r'(\d+)\s*÷\s*(\d+)', r'\1 ÷ \2', processed_text)

    # Handle temperature degrees consistently
    processed_text = re.sub(r'(\d+)\s*°\s*([CF])', r'\1°\2', processed_text)

    # Handle common unit superscripts (m², km², cm²)
    for unit in ['m', 'km', 'cm']:
        processed_text = re.sub(f'{unit}2(?![0-9])', f'{unit}²', processed_text)
        processed_text = re.sub(f'{unit}3(?![0-9])', f'{unit}³', processed_text)

    # Apply standardizations for less common notations
    for old, new in standardizations.items():
        processed_text = processed_text.replace(old, new)

    return processed_text


def process_dataset(input_file, output_file):
    """
    Process the instruction dataset and save to a new file
    """
    try:
        # Read the input JSON file
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Process each entry in the dataset
        processed_data = []
        for entry in data:
            processed_entry = {
                'instruction': standardize_notation(entry['instruction']),
                'input': standardize_notation(entry['input']),
                'output': standardize_notation(entry['output'])
            }
            processed_data.append(processed_entry)

        # Save the processed data to a new JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=4, ensure_ascii=False)

        print(f"Successfully processed {len(processed_data)} entries.")
        print(f"Processed data saved to: {output_file}")

    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{input_file}'.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


# Example usage
if __name__ == "__main__":
    input_file = "instructions.json"
    output_file = "processed_instruction_dataset.json"
    process_dataset(input_file, output_file)