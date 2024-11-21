import json
import argparse
from rwkv.model import RWKV

def test(data_dir, model_dir):
    # Load test data
    test_file = f"{data_dir}/test.json"
    with open(test_file, 'r') as f:
        testing_data = json.load(f)

    # Load model
    model_path = f"{model_dir}/rwkv_model.pth"
    model = RWKV.load(model_path)

    # Evaluate
    print("Testing RWKV model...")
    for item in testing_data:
        transcription = item["input"]
        expected_chat = item["output"]
        prediction = model.generate(transcription)
        print(f"Input: {transcription}\nExpected: {expected_chat}\nGenerated: {prediction}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Directory for testing data")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory for model")
    args = parser.parse_args()

    test(args.data_dir, args.model_dir)
