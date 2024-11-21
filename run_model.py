import argparse
from rwkv.model import RWKV

def run(model_dir):
    # Load model
    model_path = f"{model_dir}/rwkv_model.pth"
    model = RWKV.load(model_path)

    print("RWKV model ready. Enter audio transcription:")
    while True:
        transcription = input("Transcription: ")
        if transcription.lower() == "exit":
            break
        generated_chat = model.generate(transcription)
        print("Generated Chat:", generated_chat)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="Directory for model")
    args = parser.parse_args()

    run(args.model_dir)
