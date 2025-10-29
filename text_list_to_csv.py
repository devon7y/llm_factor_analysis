#!/usr/bin/env python3

# Convert a list of text to a CSV file

import csv

def parse_constructs(input_file, output_file):
    """
    Parse constructs from newline-separated file and write to CSV with one word per row.
    """
    # Read the file
    print(f"Reading {input_file}...")
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Split by newlines and clean up each word
    print("Parsing words...")
    words = []
    raw_words = content.split('\n')

    for word in raw_words:
        # Strip whitespace
        cleaned = word.strip()
        # Remove any trailing RTF artifacts like braces
        cleaned = cleaned.rstrip('{}')
        if cleaned:  # Skip empty strings
            # Replace underscores with spaces
            cleaned = cleaned.replace('_', ' ')
            # Capitalize the first letter
            cleaned = cleaned[0].upper() + cleaned[1:] if len(cleaned) > 0 else cleaned
            words.append(cleaned)

    # Remove duplicates while preserving order
    print(f"Found {len(words)} total words")
    unique_words = list(dict.fromkeys(words))
    print(f"Found {len(unique_words)} unique words")

    # Write to CSV
    print(f"Writing to {output_file}...")
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['word'])  # Header (matching wordfreq_top60000.csv structure)
        for word in unique_words:
            writer.writerow([word])

    print(f"Done! Wrote {len(unique_words)} constructs to {output_file}")

    # Print first 10 words as a sample
    print("\nFirst 10 constructs:")
    for i, word in enumerate(unique_words[:10], 1):
        print(f"  {i}. {word}")

    # Print last 10 words as a sample
    print("\nLast 10 constructs:")
    for i, word in enumerate(unique_words[-10:], len(unique_words) - 9):
        print(f"  {i}. {word}")

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        # Default behavior
        input_file = "/Users/devon7y/VS Code/LLM Factor Analysis/word_lists/constructs.txt"
        output_file = "/Users/devon7y/VS Code/LLM Factor Analysis/word_lists/constructs.csv"
    else:
        # Accept input file as argument
        input_file = sys.argv[1]
        # Generate output filename (replace .txt with .csv)
        if input_file.endswith('.txt'):
            output_file = input_file.replace('.txt', '.csv')
        else:
            output_file = input_file + '.csv'

    parse_constructs(input_file, output_file)
