import pandas as pd
import re
import sys

def clean_construct(text):
    """
    Remove abbreviations in brackets and extra spaces.
    Example: "Autism spectrum disorder (ASD)" -> "Autism spectrum disorder"
    """
    # Handle non-string values (like NaN)
    if not isinstance(text, str):
        return None

    # Remove patterns like " (ASD)" or "(ASD)"
    cleaned = re.sub(r'\s*\([^)]*\)', '', text)
    # Remove any extra whitespace
    cleaned = ' '.join(cleaned.split())
    return cleaned if cleaned else None

def combine_constructs(input_files, output_file):
    """
    Combine multiple construct CSV files, remove duplicates and clean abbreviations.

    Args:
        input_files: List of input CSV file paths
        output_file: Output CSV file path
    """
    all_constructs = []

    # Read all input files
    for file in input_files:
        print(f"Reading {file}...")
        df = pd.read_csv(file)
        # Assuming the column name is 'word' based on text_list_to_csv.py
        constructs = df['word'].tolist()
        all_constructs.extend(constructs)

    print(f"Found {len(all_constructs)} total constructs")

    # Clean constructs (remove abbreviations in brackets)
    print("Cleaning constructs...")
    cleaned_constructs = [clean_construct(c) for c in all_constructs]

    # Filter out None values
    cleaned_constructs = [c for c in cleaned_constructs if c is not None]

    # Convert to lowercase
    print("Converting to lowercase...")
    cleaned_constructs = [c.lower() for c in cleaned_constructs]

    # Remove duplicates
    print("Removing duplicates...")
    unique_constructs = list(dict.fromkeys(cleaned_constructs))  # Preserves order while removing duplicates

    # Sort alphabetically
    unique_constructs.sort(key=str.lower)

    print(f"Found {len(unique_constructs)} unique constructs after cleaning and deduplication")

    # Write to output file
    print(f"Writing to {output_file}...")
    output_df = pd.DataFrame({'word': unique_constructs})
    output_df.to_csv(output_file, index=False)

    print(f"Done! Wrote {len(unique_constructs)} constructs to {output_file}")

    # Show examples of cleaned constructs
    print("\nExamples of cleaned constructs:")
    # Find some examples that had brackets
    original_with_brackets = [c for c in all_constructs if isinstance(c, str) and '(' in c][:5]
    if original_with_brackets:
        print("Before -> After:")
        for orig in original_with_brackets:
            cleaned = clean_construct(orig)
            print(f"  {orig} -> {cleaned}")

    print("\nFirst 10 constructs:")
    for i, word in enumerate(unique_constructs[:10], 1):
        print(f"  {i}. {word}")

    print("\nLast 10 constructs:")
    for i, word in enumerate(unique_constructs[-10:], len(unique_constructs) - 9):
        print(f"  {i}. {word}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default behavior: combine constructs.csv and constructs2.csv
        input_files = [
            "/Users/devon7y/VS Code/LLM Factor Analysis/word_lists/constructs.csv",
            "/Users/devon7y/VS Code/LLM Factor Analysis/word_lists/constructs2.csv"
        ]
        output_file = "/Users/devon7y/VS Code/LLM Factor Analysis/word_lists/combined_constructs.csv"
    else:
        # Accept multiple input files as arguments
        # Last argument is the output file
        if len(sys.argv) < 3:
            print("Usage: python combine_constructs.py <input1.csv> [input2.csv ...] <output.csv>")
            sys.exit(1)
        input_files = sys.argv[1:-1]
        output_file = sys.argv[-1]

    combine_constructs(input_files, output_file)
