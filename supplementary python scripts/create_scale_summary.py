#!/usr/bin/env python3
"""
Create a summary CSV of all scales with their metadata.
"""

import pandas as pd
import os
from pathlib import Path
import re

def count_csv_rows(filepath):
    """Count rows in a CSV file (excluding header)."""
    try:
        df = pd.read_csv(filepath)
        return len(df)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def get_scale_info_from_items(items_path):
    """Extract item count, factor count, and reverse item count from items file."""
    try:
        df = pd.read_csv(items_path)

        items_count = len(df)

        # Count unique factors
        if 'factor' in df.columns:
            factors_count = df['factor'].nunique()
        else:
            factors_count = None

        # Count reverse-scored items (scoring = -1)
        if 'scoring' in df.columns:
            reverse_count = (df['scoring'] == -1).sum()
        else:
            reverse_count = None

        return items_count, factors_count, reverse_count
    except Exception as e:
        print(f"Error processing {items_path}: {e}")
        return None, None, None

def extract_full_name_from_codebook(codebook_path, scale_abbr):
    """Attempt to extract full scale name from codebook."""
    # Manual mapping based on codebook content
    full_names = {
        '16PF': '16 Personality Factors',
        '431PTQ': 'Procrastination Tendency Questionnaire',
        'AMBI': 'Analog to Multiple Broadband Inventories',
        'BAI': 'Beck Anxiety Inventory',
        'BDI': 'Beck Depression Inventory',
        'Big5': 'Big Five Personality Test (IPIP)',
        'Big5FM': 'Big Five Factor Markers (IPIP)',
        'CFCS': 'Consideration of Future Consequences Scale',
        'CIS': 'COVID-19 Impact Scale',
        'DASS': 'Depression Anxiety Stress Scales',
        'DGS': 'Duckworth Grit Scale',
        'ECR': 'Experiences in Close Relationships',
        'EPQ': 'Eysenck Personality Questionnaire',
        'EQSQ': 'Empathy Quotient and Systemizing Quotient',
        'ERRI': 'Event-Related Rumination Inventory',
        'FBPS': 'Firstborn Personality Scale',
        'FTI': 'Fisher Temperament Inventory',
        'GCB': 'Generic Conspiracist Beliefs Scale',
        'GSE': 'General Self-Efficacy Scale',
        'HEXACO': 'HEXACO Personality Inventory',
        'HSNDD': 'Hypersensitive Narcissism Scale and The Dirty Dozen',
        'HSQ': 'Humor Styles Questionnaire',
        'IRI': 'Interpersonal Reactivity Index',
        'KIMS': 'Kentucky Inventory of Mindfulness Skills',
        'LLMD12': 'LLM Dependency Scale',
        'MACHIV': 'MACH-IV (Machiavellianism)',
        'MFQ': 'Moral Foundations Questionnaire',
        'NFC': 'Need for Cognition',
        'NIS': 'Nonverbal Immediacy Scale',
        'NPAS': 'Nerdy Personality Attributes Scale',
        'OSRI': 'Open Sex Role Inventory',
        'RIASEC': 'Holland Occupational Themes',
        'RSE': 'Rosenberg Self-Esteem Scale',
        'RWAS': 'Right-Wing Authoritarianism Scale',
        'SD3': 'Short Dark Triad',
        'TMA': 'Taylor Manifest Anxiety Scale'
    }

    return full_names.get(scale_abbr, scale_abbr)

def main():
    # Directory paths
    items_dir = Path('scale_items')
    responses_dir = Path('scale_responses')
    codebooks_dir = Path('scale_codebooks')

    # Get all scale names from items directory
    scale_files = sorted(items_dir.glob('*_items.csv'))

    # Prepare data for summary
    summary_data = []

    for items_file in scale_files:
        # Extract scale abbreviation from filename
        scale_abbr = items_file.stem.replace('_items', '')

        print(f"Processing {scale_abbr}...")

        # Get item info
        items_count, factors_count, reverse_count = get_scale_info_from_items(items_file)

        # Get participant count from response data
        response_file = responses_dir / f"{scale_abbr}_data.csv"
        if response_file.exists():
            participants = count_csv_rows(response_file)
        else:
            participants = None
            print(f"  Warning: No response file found for {scale_abbr}")

        # Get full name (use mapping even if codebook doesn't exist)
        codebook_files = list(codebooks_dir.glob(f"{scale_abbr}_codebook.*"))
        full_name = extract_full_name_from_codebook(None, scale_abbr)
        if not codebook_files:
            print(f"  Warning: No codebook found for {scale_abbr}")

        # Add to summary
        summary_data.append({
            'Scale': scale_abbr,
            'Full_Name': full_name,
            'Participants': participants,
            'Items': items_count,
            'Factors': factors_count,
            'Reverse_Items': reverse_count
        })

    # Create DataFrame and sort by participants (descending)
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('Participants', ascending=False)
    output_file = 'scale_summary.csv'
    summary_df.to_csv(output_file, index=False)

    print(f"\n✓ Summary saved to {output_file}")
    print(f"\nTotal scales: {len(summary_df)}")
    print("\nPreview:")
    print(summary_df.to_string(index=False))

if __name__ == '__main__':
    main()
