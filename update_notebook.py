import json
import uuid

with open('fixmatch.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

target_idx_20 = -1
for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'code' and any("X_unlab_aug = scale_noise(combined_X_unlab)" in line for line in cell.get('source', [])):
        target_idx_20 = i
        break

if target_idx_20 != -1:
    print(f"Found target cell at index {target_idx_20}")
    
    # Comment out cell 20
    new_source_20 = []
    for line in nb['cells'][target_idx_20]['source']:
        if not line.startswith("#"):
            new_source_20.append("# " + line)
        else:
            new_source_20.append(line)
    nb['cells'][target_idx_20]['source'] = new_source_20
    
    # Comment out cell 21 (concatenation)
    next_cell = nb['cells'][target_idx_20 + 1]
    if next_cell.get('cell_type') == 'code' and any("x_unlabeled_train = tf.concat" in line for line in next_cell.get('source', [])):
        print("Found concatenation cell next to it.")
        new_source_21 = []
        for line in next_cell['source']:
            if not line.startswith("#"):
                new_source_21.append("# " + line)
            else:
                new_source_21.append(line)
        next_cell['source'] = new_source_21
        
        # Insert a new cell
        new_cell = {
            "cell_type": "code",
            "execution_count": None,
            "id": str(uuid.uuid4()),
            "metadata": {},
            "outputs": [],
            "source": [
                "# Convert the raw unlabeled data directly to a tensor.\n",
                "# The random_crop and scale_noise augmentations will be handled \n",
                "# on-the-fly inside FixMatch.train_step.\n",
                "x_unlabeled_train = tf.convert_to_tensor(combined_X_unlab, dtype=tf.float32)\n",
                "print(\"Unlabeled train tensor shape:\", x_unlabeled_train.shape)"
            ]
        }
        nb['cells'].insert(target_idx_20 + 2, new_cell)
        
        with open('fixmatch.ipynb', 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print("Successfully updated notebook.")
    else:
        print("Could not find the concatenation cell immediately after.")
else:
    print("Could not find cell 20.")
