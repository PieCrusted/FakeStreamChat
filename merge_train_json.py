import json
import glob
import os

def merge_train_json(input_dir, output_file):
    """
    Merges all JSON files in input_dir into a single train.json, 
    handling cases where one input has multiple outputs by creating separate 
    1 input to 1 output pairs.
    """
    files = glob.glob(f"{input_dir}/custom/*.json")
    merged_data = []

    for file in files:
        with open(file, 'r') as f:
            data = json.load(f)
            for item in data:
                input_text = item.get("input")
                outputs = item.get("output")

                if isinstance(outputs, list):
                    for output_text in outputs:
                        merged_data.append({"input": input_text, "output": output_text})
                else:
                     merged_data.append({"input": input_text, "output": outputs})

    # Write to train.json
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(merged_data, f, indent=2)
    
    print(f"Merged and processed {len(files)} files into {output_file}")

if __name__ == "__main__":
    merge_train_json(
        input_dir="data",
        output_file="data/train.json"
    )




