import os
import json
from datetime import datetime

# Editable Constants
MAX_LENGTH = 75 # Arbitraily choosing 75 for 10 seconds, adjust for recording length
MAX_FREQ = 0.5

def read_text_files(directory):
    """
    Read all .txt files in the specified directory and return their contents with file names.
    """
    texts = []
    for file_name in sorted(os.listdir(directory)):
        if file_name.endswith(".txt"):
            file_path = os.path.join(directory, file_name)
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read().strip()
                texts.append((file_name, content))
    return texts


def qualifier(text, file_name, stats):
    """
    Determine if a transcription is valid for inclusion as an input.
    - Reject if empty.
    - Reject if excessively long (>75 words).
    - Reject if words/phrases are excessively repeated.
    """
    if not text:  # Reject empty text
        stats["empty"].append({"file_name": file_name, "content": text})
        return False

    words = text.split()
    if len(words) > MAX_LENGTH:  # Reject excessively long transcriptions
        stats["excessively_long"].append({"file_name": file_name, "content": text})
        return False

    # Check for excessive repetition (simple heuristic: more than 50% of words are the same)
    word_counts = {}
    for word in words:
        word_counts[word] = word_counts.get(word, 0) + 1
    most_frequent_word = max(word_counts.values())
    if most_frequent_word > len(words) * MAX_FREQ:
        stats["repeated_words"].append({"file_name": file_name, "content": text})
        return False

    return True


def write_json(data, output_directory, file_name):
    """
    Write the data into a JSON file in the specified output directory.
    """
    os.makedirs(output_directory, exist_ok=True)
    file_path = os.path.join(output_directory, file_name)
    with open(file_path, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)
    print(f"JSON file written to {file_path}")


def process_transcriptions(input_directory, output_directory):
    """
    Process all text files in the input directory and write qualified inputs to a JSON file.
    Also, write rejected transcriptions to a separate JSON file.
    """
    texts = read_text_files(input_directory)
    total_files = len(texts)
    processed_data = []
    rejection_stats = {
        "empty": [],
        "excessively_long": [],
        "repeated_words": [],
    }

    for file_name, text in texts:
        if qualifier(text, file_name, rejection_stats):
            processed_data.append({
                "input": text,
                "output": []
            })

    # Write accepted inputs to JSON
    accepted_file_name = datetime.now().strftime("%Y-%m-%d.json")
    write_json(processed_data, output_directory, accepted_file_name)

    # Write rejected transcriptions to JSON
    rejected_file_name = datetime.now().strftime("%Y-%m-%d_rejected.json")
    rejected_data = {
        "empty_transcriptions": rejection_stats["empty"],
        "excessively_long_transcriptions": rejection_stats["excessively_long"],
        "repeated_words_transcriptions": rejection_stats["repeated_words"],
    }
    write_json(rejected_data, output_directory, rejected_file_name)

    # Calculate acceptance and rejection rates
    accepted_files = len(processed_data)
    rejected_files = total_files - accepted_files
    acceptance_rate = (accepted_files / total_files) * 100 if total_files > 0 else 0
    rejection_rate = (rejected_files / total_files) * 100 if total_files > 0 else 0

    # Print statistics
    print("\nProcessing Summary:")
    print(f"Total files processed: {total_files}")
    print(f"Accepted files: {accepted_files} ({acceptance_rate:.2f}%)")
    print(f"Rejected files: {rejected_files} ({rejection_rate:.2f}%)")
    print("\nRejection Statistics:")
    print(f"Empty transcriptions: {len(rejection_stats['empty'])}")
    print(f"Excessively long transcriptions: {len(rejection_stats['excessively_long'])}")
    print(f"Transcriptions with repeated words: {len(rejection_stats['repeated_words'])}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process transcription text files into JSON format.")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the directory containing transcription .txt files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the directory where the output JSON file should be saved.",
    )
    args = parser.parse_args()

    process_transcriptions(args.input_dir, args.output_dir)
