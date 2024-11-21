import json
import argparse
from rwkv.model import RWKV
from rwkv.trainer import Trainer

# Training script
def train(data_dir, model_dir, checkpoint_dir):
    # Load training data
    train_file = f"{data_dir}/train.json"
    with open(train_file, 'r') as f:
        training_data = json.load(f)

    # Prepare model
    model_path = f"{model_dir}/rwkv_model.pth"
    trainer = Trainer(
        model_path=model_path,
        checkpoint_dir=checkpoint_dir,
        epochs=5,  # Adjust as needed
        batch_size=16
    )

    # Training loop
    print("Training RWKV model...")
    trainer.train(training_data)
    print("Training complete. Model saved at:", model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Directory for training data")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory to save trained model")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Directory to save checkpoints")
    args = parser.parse_args()

    train(args.data_dir, args.model_dir, args.checkpoint_dir)
