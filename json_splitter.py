import os
import json
from datetime import datetime

# Constants
INPUT_DIRECTORY = "json_splitter/"
OUTPUT_DIRECTORY = "json_splitter/split/"
SPLIT_SIZE = 50  # Number of items per split file

# Ensure output directory exists
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

def split_json_files():
    # Gather all JSON files in the input directory
    json_files = [f for f in os.listdir(INPUT_DIRECTORY) if f.endswith('.json')]

    for json_file in json_files:
        file_path = os.path.join(INPUT_DIRECTORY, json_file)

        with open(file_path, 'r') as file:
            try:
                data = json.load(file)
                if not isinstance(data, list):
                    print(f"Skipping {json_file}: Expected a list at the root level.")
                    continue
            except json.JSONDecodeError as e:
                print(f"Error reading {json_file}: {e}")
                continue

        # Split data into chunks
        base_name = os.path.splitext(json_file)[0]  # Get the base name of the file without extension
        date_str = datetime.now().strftime("%m-%d-%Y")

        for i in range(0, len(data), SPLIT_SIZE):
            chunk = data[i:i + SPLIT_SIZE]
            part_number = (i // SPLIT_SIZE) + 1
            output_file_name = f"{base_name}_{date_str}-part{part_number}.json"
            output_file_path = os.path.join(OUTPUT_DIRECTORY, output_file_name)

            with open(output_file_path, 'w') as output_file:
                json.dump(chunk, output_file, indent=4)

            print(f"Written: {output_file_path}")

if __name__ == "__main__":
    split_json_files()
