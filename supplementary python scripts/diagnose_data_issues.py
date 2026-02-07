#!/usr/bin/env python3
"""
Diagnose data quality issues that can cause EFA to fail.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

def diagnose_scale_data(scale_name):
    """Diagnose potential issues with scale data."""

    print(f"\n{'='*70}")
    print(f"DIAGNOSING: {scale_name}")
    print(f"{'='*70}\n")

    # Load items file
    items_file = Path(f"scale_items/{scale_name}_items.csv")
    if not items_file.exists():
        print(f"❌ Items file not found: {items_file}")
        return

    items_df = pd.read_csv(items_file)
    codes = items_df['code'].tolist()

    # Load response data
    data_file = Path(f"scale_responses/{scale_name}_data.csv")
    if not data_file.exists():
        print(f"❌ Response data file not found: {data_file}")
        return

    data = pd.read_csv(data_file)

    # Check if all item codes exist in data
    missing_codes = [code for code in codes if code not in data.columns]
    if missing_codes:
        print(f"⚠️  Warning: {len(missing_codes)} item codes not found in data:")
        print(f"   {missing_codes[:10]}{'...' if len(missing_codes) > 10 else ''}")
        # Filter to only codes that exist
        codes = [c for c in codes if c in data.columns]
        print(f"   Using {len(codes)} codes that exist in data\n")

    # Select only the item columns
    item_data = data[codes]

    print(f"[1] Dataset Shape")
    print(f"    Participants: {len(item_data)}")
    print(f"    Items: {len(codes)}")

    # Check for missing values
    print(f"\n[2] Missing Values")
    missing_counts = item_data.isnull().sum()
    if missing_counts.sum() == 0:
        print(f"    ✓ No missing values")
    else:
        n_items_with_missing = (missing_counts > 0).sum()
        print(f"    ⚠️  {n_items_with_missing} items have missing values:")
        top_missing = missing_counts[missing_counts > 0].sort_values(ascending=False).head(10)
        for item, count in top_missing.items():
            pct = (count / len(item_data)) * 100
            print(f"       {item}: {count} ({pct:.1f}%)")

    # Check for zero/low variance items
    print(f"\n[3] Variance Check")
    variances = item_data.var()
    zero_var = variances[variances == 0]
    low_var = variances[(variances > 0) & (variances < 0.01)]

    if len(zero_var) > 0:
        print(f"    ❌ {len(zero_var)} items have ZERO variance (all same value):")
        for item in zero_var.index[:10]:
            unique_val = item_data[item].dropna().unique()
            print(f"       {item}: value = {unique_val}")
    else:
        print(f"    ✓ No zero-variance items")

    if len(low_var) > 0:
        print(f"    ⚠️  {len(low_var)} items have very LOW variance:")
        for item in low_var.index[:10]:
            print(f"       {item}: var = {low_var[item]:.6f}")

    # Check for constant items
    print(f"\n[4] Unique Values per Item")
    unique_counts = item_data.nunique()
    single_value = unique_counts[unique_counts <= 1]
    very_few_values = unique_counts[(unique_counts > 1) & (unique_counts <= 2)]

    if len(single_value) > 0:
        print(f"    ❌ {len(single_value)} items have only 1 unique value:")
        for item in single_value.index[:10]:
            print(f"       {item}")
    else:
        print(f"    ✓ No single-value items")

    if len(very_few_values) > 0:
        print(f"    ⚠️  {len(very_few_values)} items have only 2 unique values:")
        for item in very_few_values.index[:10]:
            print(f"       {item}: {sorted(item_data[item].dropna().unique())}")

    # Check for perfect correlations
    print(f"\n[5] Correlation Matrix")
    try:
        # Drop rows with any missing values for correlation calculation
        clean_data = item_data.dropna()
        print(f"    Using {len(clean_data)} complete cases for correlation")

        if len(clean_data) < 10:
            print(f"    ❌ Too few complete cases ({len(clean_data)}) to calculate correlations")
            return

        corr_matrix = clean_data.corr()

        # Check for perfect correlations (excluding diagonal)
        np.fill_diagonal(corr_matrix.values, 0)
        perfect_corr = np.abs(corr_matrix) >= 0.999

        if perfect_corr.sum().sum() > 0:
            print(f"    ⚠️  Found {perfect_corr.sum().sum() // 2} pairs with correlation >= 0.999:")
            pairs_found = 0
            for i in range(len(corr_matrix)):
                for j in range(i+1, len(corr_matrix)):
                    if perfect_corr.iloc[i, j]:
                        print(f"       {corr_matrix.index[i]} <-> {corr_matrix.columns[j]}: r = {corr_matrix.iloc[i, j]:.6f}")
                        pairs_found += 1
                        if pairs_found >= 10:
                            break
                if pairs_found >= 10:
                    print(f"       ... (showing first 10 pairs)")
                    break
        else:
            print(f"    ✓ No perfect correlations")

        # Check matrix properties
        det = np.linalg.det(corr_matrix)
        print(f"    Determinant: {det:.2e}")
        if abs(det) < 1e-10:
            print(f"    ❌ Matrix is singular or near-singular (det ≈ 0)")
        else:
            print(f"    ✓ Matrix is non-singular")

        # Check condition number
        cond = np.linalg.cond(corr_matrix)
        print(f"    Condition number: {cond:.2e}")
        if cond > 1e10:
            print(f"    ⚠️  Matrix is poorly conditioned (high condition number)")
        else:
            print(f"    ✓ Matrix is well-conditioned")

    except Exception as e:
        print(f"    ❌ Error calculating correlations: {e}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    issues = []
    if len(zero_var) > 0:
        issues.append(f"❌ {len(zero_var)} zero-variance items")
    if len(single_value) > 0:
        issues.append(f"❌ {len(single_value)} single-value items")
    if missing_counts.sum() > 0:
        issues.append(f"⚠️  Missing values in {n_items_with_missing} items")
    if 'det' in locals() and abs(det) < 1e-10:
        issues.append(f"❌ Singular correlation matrix")

    if issues:
        print("Issues found:")
        for issue in issues:
            print(f"  {issue}")
        print("\nRecommendations:")
        if len(zero_var) > 0 or len(single_value) > 0:
            print("  1. Remove items with zero variance or single values")
        if missing_counts.sum() > 0:
            print("  2. Handle missing values (remove participants or impute)")
        if 'det' in locals() and abs(det) < 1e-10:
            print("  3. Check for and remove perfectly correlated items")
    else:
        print("✓ No major issues detected!")

    print()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        scale_name = sys.argv[1]
        diagnose_scale_data(scale_name)
    else:
        print("Usage: python diagnose_data_issues.py SCALE_NAME")
        print("\nExample: python diagnose_data_issues.py DASS")
