import json
import glob
import os
import random
import argparse

def merge_and_split_json_files(input_dir, train_file, test_file, test_split=0.2):
    """
    Merges JSON files, splits them into train and test sets, 
    and handles cases where one input has multiple outputs by creating separate
    input-output pairs.
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


    # Shuffle data for random splitting
    random.shuffle(merged_data)

    # Split data
    split_idx = int(len(merged_data) * (1 - test_split))
    train_data = merged_data[:split_idx]
    test_data = merged_data[split_idx:]

    # Write to files
    os.makedirs(os.path.dirname(train_file), exist_ok=True)
    with open(train_file, 'w') as f:
        json.dump(train_data, f, indent=2)
    with open(test_file, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"Split {len(merged_data)} items into {len(train_data)} train and {len(test_data)} test examples.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge JSON files and optionally split into train/test sets.")
    parser.add_argument("--test_split", type=float, default=0.2, help="Proportion of data to use for testing (default: 0.2).")
    args = parser.parse_args()

    if args.test_split < 0 or args.test_split > 1.0:
        raise ValueError("test_split must be between 0 and 1.0")

    merge_and_split_json_files(
        input_dir="data",
        train_file="data/train.json",
        test_file="data/test.json",
        test_split=args.test_split
    )
