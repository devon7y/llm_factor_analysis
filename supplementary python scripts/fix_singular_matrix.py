#!/usr/bin/env python3
"""
Helper functions to fix singular matrix issues in EFA.
"""

import numpy as np
import pandas as pd
from scipy import linalg

def check_matrix_singularity(corr_matrix, verbose=True):
    """Check if correlation matrix is singular or near-singular."""
    det = np.linalg.det(corr_matrix)
    cond = np.linalg.cond(corr_matrix)

    if verbose:
        print(f"  Matrix determinant: {det:.2e}")
        print(f"  Condition number: {cond:.2e}")

    is_singular = abs(det) < 1e-10
    is_ill_conditioned = cond > 1e10

    return is_singular, is_ill_conditioned, det, cond

def regularize_correlation_matrix(corr_matrix, alpha=1e-6, verbose=True):
    """
    Add small regularization to correlation matrix to make it non-singular.

    This adds alpha to the diagonal, which is equivalent to ridge regularization.

    Parameters:
    -----------
    corr_matrix : pd.DataFrame or np.ndarray
        The correlation matrix
    alpha : float
        Regularization parameter (default: 1e-6)
    verbose : bool
        Print information about the regularization

    Returns:
    --------
    regularized_corr : pd.DataFrame or np.ndarray
        Regularized correlation matrix
    """
    is_dataframe = isinstance(corr_matrix, pd.DataFrame)

    if is_dataframe:
        index = corr_matrix.index
        columns = corr_matrix.columns
        corr_array = corr_matrix.values
    else:
        corr_array = corr_matrix

    # Add regularization to diagonal
    n = corr_array.shape[0]
    regularized = corr_array + alpha * np.eye(n)

    # Renormalize diagonal to 1
    diag = np.sqrt(np.diag(regularized))
    regularized = regularized / diag[:, None] / diag[None, :]

    if verbose:
        det_before = np.linalg.det(corr_array)
        det_after = np.linalg.det(regularized)
        print(f"  Regularization applied (alpha={alpha})")
        print(f"  Determinant before: {det_before:.2e}")
        print(f"  Determinant after: {det_after:.2e}")

    if is_dataframe:
        return pd.DataFrame(regularized, index=index, columns=columns)
    else:
        return regularized

def preprocess_data_for_efa(data, min_variance=0.001, remove_missing=True,
                            check_correlations=True, verbose=True):
    """
    Preprocess data to avoid EFA failures.

    Parameters:
    -----------
    data : pd.DataFrame
        Response data with items as columns
    min_variance : float
        Minimum variance threshold for items
    remove_missing : bool
        Whether to remove rows with missing values
    check_correlations : bool
        Whether to check for perfect correlations
    verbose : bool
        Print diagnostic information

    Returns:
    --------
    clean_data : pd.DataFrame
        Preprocessed data
    removed_items : list
        List of removed item codes
    """
    if verbose:
        print(f"\n{'='*70}")
        print("DATA PREPROCESSING FOR EFA")
        print(f"{'='*70}\n")
        print(f"Initial data shape: {data.shape}")

    clean_data = data.copy()
    removed_items = []

    # 1. Remove items with low/zero variance
    variances = clean_data.var()
    low_var_items = variances[variances < min_variance].index.tolist()

    if len(low_var_items) > 0:
        if verbose:
            print(f"\n[1] Removing {len(low_var_items)} low-variance items:")
            for item in low_var_items[:10]:
                print(f"    {item}: variance = {variances[item]:.6f}")
            if len(low_var_items) > 10:
                print(f"    ... and {len(low_var_items) - 10} more")
        clean_data = clean_data.drop(columns=low_var_items)
        removed_items.extend(low_var_items)
    elif verbose:
        print(f"\n[1] ✓ No low-variance items")

    # 2. Remove rows with missing values
    if remove_missing:
        n_before = len(clean_data)
        clean_data = clean_data.dropna()
        n_after = len(clean_data)
        n_removed = n_before - n_after

        if verbose:
            if n_removed > 0:
                print(f"\n[2] Removed {n_removed} rows with missing values ({n_removed/n_before*100:.1f}%)")
            else:
                print(f"\n[2] ✓ No missing values")

    # 3. Check for perfect correlations
    if check_correlations and len(clean_data) > 0:
        if verbose:
            print(f"\n[3] Checking for perfect correlations...")

        corr_matrix = clean_data.corr()
        np.fill_diagonal(corr_matrix.values, 0)
        perfect_corr = np.abs(corr_matrix) >= 0.999

        if perfect_corr.sum().sum() > 0:
            # Find pairs and remove one from each pair
            pairs_to_remove = set()
            for i in range(len(corr_matrix)):
                for j in range(i+1, len(corr_matrix)):
                    if perfect_corr.iloc[i, j]:
                        item_i = corr_matrix.index[i]
                        item_j = corr_matrix.columns[j]
                        if verbose:
                            print(f"    Found perfect correlation: {item_i} <-> {item_j} (r={corr_matrix.iloc[i, j]:.6f})")
                        # Keep the first item, remove the second
                        pairs_to_remove.add(item_j)

            if len(pairs_to_remove) > 0:
                if verbose:
                    print(f"    Removing {len(pairs_to_remove)} items from perfect correlation pairs")
                clean_data = clean_data.drop(columns=list(pairs_to_remove))
                removed_items.extend(list(pairs_to_remove))
        elif verbose:
            print(f"    ✓ No perfect correlations found")

    if verbose:
        print(f"\n{'='*70}")
        print(f"Final data shape: {clean_data.shape}")
        if len(removed_items) > 0:
            print(f"Total removed items: {len(removed_items)}")
        print(f"{'='*70}\n")

    return clean_data, removed_items

def safe_calculate_kmo(corr_matrix, regularize_if_needed=True, alpha=1e-6, verbose=True):
    """
    Safely calculate KMO, with automatic regularization if needed.

    Parameters:
    -----------
    corr_matrix : pd.DataFrame or np.ndarray
        Correlation matrix
    regularize_if_needed : bool
        Whether to apply regularization if matrix is singular
    alpha : float
        Regularization parameter
    verbose : bool
        Print diagnostic information

    Returns:
    --------
    kmo_per_item : array-like
        KMO value for each item
    kmo_total : float
        Overall KMO value
    """
    from factor_analyzer.factor_analyzer import calculate_kmo

    try:
        # Try standard calculation first
        kmo_per_item, kmo_total = calculate_kmo(corr_matrix)
        if verbose:
            print(f"  ✓ KMO calculated successfully")
        return kmo_per_item, kmo_total

    except (np.linalg.LinAlgError, AssertionError) as e:
        if verbose:
            print(f"  ⚠️  KMO calculation failed: {e}")

        if regularize_if_needed:
            if verbose:
                print(f"  Attempting with regularization...")

            # Regularize and try again
            reg_corr = regularize_correlation_matrix(corr_matrix, alpha=alpha, verbose=verbose)

            try:
                kmo_per_item, kmo_total = calculate_kmo(reg_corr)
                if verbose:
                    print(f"  ✓ KMO calculated successfully with regularization")
                return kmo_per_item, kmo_total
            except Exception as e2:
                if verbose:
                    print(f"  ❌ KMO calculation failed even with regularization: {e2}")
                raise
        else:
            raise

def safe_calculate_bartlett(corr_matrix, n_samples, regularize_if_needed=True,
                           alpha=1e-6, verbose=True):
    """
    Safely calculate Bartlett's test, with automatic regularization if needed.

    Parameters:
    -----------
    corr_matrix : pd.DataFrame or np.ndarray
        Correlation matrix
    n_samples : int
        Number of samples
    regularize_if_needed : bool
        Whether to apply regularization if matrix is singular
    alpha : float
        Regularization parameter
    verbose : bool
        Print diagnostic information

    Returns:
    --------
    chi2 : float
        Chi-square statistic
    p_value : float
        P-value
    """
    from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity

    try:
        # Try standard calculation first
        chi2, p_value = calculate_bartlett_sphericity(corr_matrix, n_samples)
        if verbose:
            print(f"  ✓ Bartlett test calculated successfully")
        return chi2, p_value

    except (np.linalg.LinAlgError, AssertionError) as e:
        if verbose:
            print(f"  ⚠️  Bartlett test failed: {e}")

        if regularize_if_needed:
            if verbose:
                print(f"  Attempting with regularization...")

            # Regularize and try again
            reg_corr = regularize_correlation_matrix(corr_matrix, alpha=alpha, verbose=verbose)

            try:
                chi2, p_value = calculate_bartlett_sphericity(reg_corr, n_samples)
                if verbose:
                    print(f"  ✓ Bartlett test calculated successfully with regularization")
                return chi2, p_value
            except Exception as e2:
                if verbose:
                    print(f"  ❌ Bartlett test failed even with regularization: {e2}")
                raise
        else:
            raise
