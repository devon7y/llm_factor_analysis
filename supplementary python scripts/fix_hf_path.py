import json
import os

nb_path = '/Users/devon7y/VS Code/LLM_Factor_Analysis/qwen3_efa.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# We look for the cell containing the path assignment
target_str = "/home/devon7y/links/scratch/huggingface"
replacement_str = os.path.expanduser("~/.cache/huggingface")

found = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        modified_cell = False
        for line in cell['source']:
            if target_str in line:
                # Replace the specific path string
                # We need to be careful about the quotes and escaping in the JSON source list
                # The line in JSON source list usually ends with \n
                new_line = line.replace(target_str, replacement_str)
                new_source.append(new_line)
                modified_cell = True
                found = True
            else:
                new_source.append(line)
        
        if modified_cell:
            cell['source'] = new_source

if found:
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Notebook paths fixed successfully.")
else:
    print("Target path not found.")
