import json
import glob
import os

def merge_train_json(input_dir, output_file):
    """Merge all JSON files in input_dir into a single train.json."""
    files = glob.glob(f"{input_dir}/custom/*.json")
    merged_data = []

    for file in files:
        with open(file, 'r') as f:
            data = json.load(f)
            merged_data.extend(data)

    # Write to train.json
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(merged_data, f, indent=2)
    
    print(f"Merged {len(files)} files into {output_file}")

if __name__ == "__main__":
    merge_train_json(
        input_dir="data",
        output_file="data/train.json"
    )
