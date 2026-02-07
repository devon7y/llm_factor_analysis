import json
import os

nb_path = '/Users/devon7y/VS Code/LLM_Factor_Analysis/qwen3_efa.ipynb'

with open(nb_path, 'r') as f:
    nb = json.load(f)

new_source = [
    "# Robust patching to prevent recursion\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import factor_analyzer.factor_analyzer as fa\n",
    "import importlib\n",
    "\n",
    "# Ensure we have the original function\n",
    "if not hasattr(fa, '_original_calculate_kmo'):\n",
    "    # Reload to get clean state if needed\n",
    "    importlib.reload(fa)\n",
    "    fa._original_calculate_kmo = fa.calculate_kmo\n",
    "    fa._original_calculate_bartlett = fa.calculate_bartlett_sphericity\n",
    "    print(\"Loaded and saved original factor_analyzer functions.\")\n",
    "else:\n",
    "    print(\"Using previously saved original factor_analyzer functions.\")\n",
    "\n",
    "def regularize_correlation_matrix(corr_matrix, alpha=1e-6):\n",
    "    \"\"\"Add regularization to correlation matrix.\"\"\"\n",
    "    is_df = isinstance(corr_matrix, pd.DataFrame)\n",
    "    corr_array = corr_matrix.values.copy() if is_df else corr_matrix.copy()\n",
    "    \n",
    "    n = corr_array.shape[0]\n",
    "    regularized = corr_array + alpha * np.eye(n)\n",
    "    diag = np.sqrt(np.diag(regularized))\n",
    "    regularized = regularized / diag[:, None] / diag[None, :]\n",
    "    \n",
    "    if is_df:\n",
    "        return pd.DataFrame(regularized, index=corr_matrix.index, columns=corr_matrix.columns)\n",
    "    return regularized\n",
    "\n",
    "def safe_calculate_kmo(corr_matrix, alpha=1e-6):\n",
    "    \"\"\"KMO with auto-regularization.\"\"\"\n",
    "    try:\n",
    "        return fa._original_calculate_kmo(corr_matrix)\n",
    "    except (np.linalg.LinAlgError, AssertionError):\n",
    "        print(f\"  ⚠️  Singular matrix detected, applying regularization (alpha={alpha})...\")\n",
    "        return fa._original_calculate_kmo(regularize_correlation_matrix(corr_matrix, alpha))\n",
    "\n",
    "def safe_calculate_bartlett(corr_matrix, alpha=1e-6):\n",
    "    \"\"\"Bartlett with auto-regularization.\"\"\"\n",
    "    try:\n",
    "        return fa._original_calculate_bartlett(corr_matrix)\n",
    "    except (np.linalg.LinAlgError, AssertionError):\n",
    "        print(f\"  ⚠️  Singular matrix detected, applying regularization (alpha={alpha})...\")\n",
    "        return fa._original_calculate_bartlett(regularize_correlation_matrix(corr_matrix, alpha))\n",
    "\n",
    "# Clean data - remove missing values (handle both DataFrame and ndarray)\n",
    "print(f\"\\n{'='*70}\")\n",
    "print(\"PREPROCESSING EMPIRICAL DATA\")\n",
    "print(f\"{'='*70}\")\n",
    "\n",
    "if isinstance(empirical_data, pd.DataFrame):\n",
    "    print(f\"Original shape: {empirical_data.shape}\")\n",
    "    n_before = len(empirical_data)\n",
    "    empirical_data = empirical_data.dropna()\n",
    "    n_after = len(empirical_data)\n",
    "    if n_before - n_after > 0:\n",
    "        print(f\"Removed {n_before - n_after} rows with missing values ({(n_before - n_after)/n_before*100:.1f}%)\")\n",
    "    else:\n",
    "        print(\"No missing values found\")\n",
    "    print(f\"Final shape: {empirical_data.shape}\")\n",
    "elif isinstance(empirical_data, np.ndarray):\n",
    "    print(f\"Original shape: {empirical_data.shape}\")\n",
    "    # For numpy arrays, remove rows with any NaN values\n",
    "    mask = ~np.isnan(empirical_data).any(axis=1)\n",
    "    n_before = len(empirical_data)\n",
    "    empirical_data = empirical_data[mask]\n",
    "    n_after = len(empirical_data)\n",
    "    if n_before - n_after > 0:\n",
    "        print(f\"Removed {n_before - n_after} rows with missing values ({(n_before - n_after)/n_before*100:.1f}%)\")\n",
    "    else:\n",
    "        print(\"No missing values found\")\n",
    "    print(f\"Final shape: {empirical_data.shape}\")\n",
    "else:\n",
    "    print(f\"Data type: {type(empirical_data)}\")\n",
    "    print(\"Warning: Could not clean data (unexpected type)\")\n",
    "\n",
    "print(f\"{'='*70}\\n\")\n",
    "\n",
    "# Patch functions globally\n",
    "fa.calculate_kmo = safe_calculate_kmo\n",
    "fa.calculate_bartlett_sphericity = safe_calculate_bartlett\n",
    "\n",
    "print(\"✓ Installed safe KMO and Bartlett calculation functions (Recursion-proof)\")\n",
    "print(\"✓ These will automatically handle singular matrices with regularization\\n\")\n"
]

found = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source_str = "".join(cell['source'])
        if "def safe_calculate_kmo" in source_str:
            cell['source'] = new_source
            found = True
            break

if found:
    with open(nb_path, 'w') as f:
        json.dump(nb, f, indent=1)
    print("Successfully patched notebook.")
else:
    print("Could not find the cell to patch.")
