import json
import os

nb_path = '/Users/devon7y/VS Code/LLM_Factor_Analysis/qwen3_efa.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

target_source_start = [
    "# ==============================================================================\n",
    "# FIX FOR SINGULAR MATRIX - AUTO-REGULARIZATION FOR KMO/BARTLETT\n",
    "# ==============================================================================\n",
    "\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity\n"
]

replacement_source_start = [
    "# ==============================================================================\n",
    "# FIX FOR SINGULAR MATRIX - AUTO-REGULARIZATION FOR KMO/BARTLETT\n",
    "# ==============================================================================\n",
    "\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import importlib\n",
    "import factor_analyzer.factor_analyzer\n",
    "importlib.reload(factor_analyzer.factor_analyzer)\n",
    "from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity\n"
]

found = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        # Check if the start of the source matches
        if len(source) >= len(target_source_start):
            match = True
            for i in range(len(target_source_start)):
                if source[i] != target_source_start[i]:
                    match = False
                    break
            
            if match:
                print("Found target cell. Applying fix...")
                # Replace the beginning of the source
                new_source = replacement_source_start + source[len(target_source_start):]
                cell['source'] = new_source
                found = True
                break

if found:
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Notebook fixed successfully.")
else:
    print("Target cell not found.")
