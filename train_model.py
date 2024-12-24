import json
import argparse
import os
import re
import gc
import types
import ssl
import requests

# Copied from the RWKV python code
os.environ['RWKV_JIT_ON'] = '1'  # '1' for better speed

from rwkv.model import RWKV
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
import certifi

# Custom Dataset Class for RWKV training
class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_seq_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = item['input']
        output_text = item['output']

        input_ids = self.tokenizer.encode(input_text).ids
        output_ids = self.tokenizer.encode(output_text).ids

        # Combine input and output IDs with a special token
        combined_ids = input_ids + [50256] + output_ids  # Special token to separate input and output

        # Truncate if it's too long
        combined_ids = combined_ids[:self.max_seq_length]

        # Pad sequence so every batch have the same length
        padding_length = self.max_seq_length - len(combined_ids)
        padded_ids = combined_ids + [0] * padding_length  # Padding token

        return torch.tensor(padded_ids[:-1], dtype=torch.long), torch.tensor(padded_ids[1:], dtype=torch.long)  # Input and target for seq2seq

def create_tokenizer(data, vocab_size=50257):
    """Creates and trains a tokenizer from a list of strings."""
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<pad>", "<unk>", "<s>", "</s>", "<sep>", "<mask>"])
    tokenizer.train_from_iterator(data, trainer=trainer, length=len(data))
    return tokenizer

# What I found by manually going to BlinkDL's HuggingFace Profile
KNOWN_MODELS = {
    # 367 MB
    "BlinkDL/rwkv-4-pile-169m" : "https://huggingface.co/BlinkDL/rwkv-4-pile-169m/resolve/main/RWKV-4b-Pile-171M-20230202-7922.pth",
    # 339 MB
    "RWKV-4-PilePlus-169M-20230520-done-ctx4096.pth" : "https://huggingface.co/BlinkDL/rwkv-4-pileplus/resolve/main/RWKV-4-PilePlus-169M-20230520-done-ctx4096.pth",
    # 861 MB
    "RWKV-4-PilePlus-430M-20230520-6162-1018Gtokens-ctx4096.pth" :"https://huggingface.co/BlinkDL/rwkv-4-pileplus/resolve/main/RWKV-4-PilePlus-430M-20230520-6162-1018Gtokens-ctx4096.pth",
    # 3.03 GB
    "RWKV-4-PilePlus-1B5-20230520-2942-486Gtokens-ctx4096.pth" : "https://huggingface.co/BlinkDL/rwkv-4-pileplus/resolve/main/RWKV-4-PilePlus-1B5-20230520-2942-486Gtokens-ctx4096.pth",
    # 5.97 GB
    "RWKV-4-PilePlus-3B-20230520-3147-520Gtokens-ctx4096.pth" : "https://huggingface.co/BlinkDL/rwkv-4-pileplus/resolve/main/RWKV-4-PilePlus-3B-20230520-3147-520Gtokens-ctx4096.pth",
    # 335 MB
    "RWKV-x070-Pile-168M-20241120-ctx4096.pth" : "https://huggingface.co/BlinkDL/rwkv-7-pile/resolve/main/RWKV-x070-Pile-168M-20241120-ctx4096.pth",
    # 842 MB
    "RWKV-x070-Pile-421M-20241127-ctx4096.pth" : "https://huggingface.co/BlinkDL/rwkv-7-pile/resolve/main/RWKV-x070-Pile-421M-20241127-ctx4096.pth"
}
def download_model(model_name, model_dir):
    if model_name in KNOWN_MODELS:
        url = KNOWN_MODELS[model_name]
        file_name = os.path.join(model_dir, f"{model_name}")
        if os.path.exists(file_name):
            print(f"Model {model_name} already downloaded at {file_name}")
            return file_name
        print(f"Downloading model {model_name} from {url} to {file_name}")
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        response = requests.get(url, stream=True, verify=certifi.where())
        response.raise_for_status() # Raise an exception for bad status codes (like 404 or 500)
        with open(file_name, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return file_name
    else:
        raise ValueError(f"Unknown model: {model_name}")


# # fp16 = good for GPU
# fp32 = good for CPU
# bf16 = supports CPU
def train(data_dir, model_dir, checkpoint_dir, model_type="RWKV-4-World", \
    strategy="cpu fp32", epochs=5, batch_size=16, vocab_size=50257, \
    max_seq_length=256, learning_rate=1e-4, load_model=None, num_layers=4, \
    embed_dim=128, hidden_dim=256, head_dim=32):
    # Load training data
    train_file = f"{data_dir}/train.json"
    try:
        with open(train_file, 'r') as f:
            training_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Training data not found at {train_file}")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {train_file}")
        return

    # Extract all text from dataset
    all_text_data = []
    for item in training_data:
        all_text_data.append(item['input'])
        all_text_data.append(item['output'])

    # Create Tokenizer
    tokenizer = create_tokenizer(all_text_data, vocab_size=vocab_size)

    # Prepare Data Loader
    dataset = TextDataset(training_data, tokenizer, max_seq_length=max_seq_length)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Prepare model
    model_path = f"{model_dir}/rwkv_model.pth"

    # Download and Load Model
    model_path = download_model(model_type, model_dir) # download the specified model, overwrites dummy file if it exists
    
    # Model Instantiation
    if load_model:
        try:
            print(f"Loading model from file {load_model}")
            loaded_state = torch.load(load_model, map_location=torch.device('cpu'))
            model = RWKV(model=load_model, strategy=strategy)
            model.load_state_dict(loaded_state, strict=False)
            print(f"Loaded weights from {load_model}")
        except FileNotFoundError:
            print(f"Warning: Pre-trained model file not found at {load_model}, loading default model {model_type}.")
            loaded_state = torch.load(model_path, map_location=torch.device('cpu'))
            model = RWKV(model=model_path, strategy=strategy)
            model.load_state_dict(loaded_state, strict=False)
        except Exception as e:
            print(f"Warning: Error loading model {e}, loading default model {model_type}")
            loaded_state = torch.load(model_path, map_location=torch.device('cpu'))
            model = RWKV(model=model_path, strategy=strategy)
            model.load_state_dict(loaded_state, strict=False)
    else:
        print(f"Loading default model {model_type}")
        loaded_state = torch.load(model_path, map_location=torch.device('cpu'))
        model = RWKV(model=model_path, strategy=strategy)
        model.load_state_dict(loaded_state, strict=False)

    # Debugging: Print parameters before optimizer
    print("Model parameters:")
    for name, param in model.named_parameters():
        print(name, type(param), param.size(), param.requires_grad)

    # Optimizer
    # if not list(model.parameters()):
    #     print("Error: Model parameters were not loaded correctly.")
    #     return  # Exit the function if no parameters are fou
    # # Check if model.parameters is not empty
    # print(f"Checking model parameters: type(model)={type(model)}")
    # if not list(model.state_dict()):
    #     print("Error: Model parameters were not loaded correctly.")
    #     # print(dir(model))
    #     # print(model.children())

    #     # print("Checking model named modules:")
    #     # for name, module in model.named_modules():
    #     #     print(f"{name}: type(module): {type(module)}")
    #     #     if hasattr(module, 'parameters'):
    #     #         for p_name, p in module.named_parameters(recurse = True):
    #     #             print(f"module {name} parameter name: {p_name} type(parameter):{type(p)} size: {p.size()} requires_grad: {p.requires_grad}")
    #     # print("Checking the type of the state_dict")
    #     # print(f"State dict type is {type(loaded_state)}")
    #     # if isinstance(loaded_state, dict):
    #     #     for name, param in loaded_state.items():
    #     #         if isinstance(param, torch.Tensor):
    #     #             print(f"State dict parameter name: {name}, type(param):{type(param)}, size: {param.size()}")
    #     #         else:
    #     #             print(f"State dict parameter name: {name}, type: {type(param)}")
    # return

    # parameters = [param for param in model.state_dict().values() if isinstance(param, torch.Tensor)]

    # Extract parameters from self.w
    parameters = []
    for key, value in model.w.items():
        if isinstance(value, torch.Tensor):
            value.requires_grad = True  # Ensure gradients are tracked
            parameters.append(value)

    # Verify the extracted parameters
    if not parameters:
        raise ValueError("No trainable parameters found in the model.")

    # Create optimizer
    optimizer = optim.AdamW(parameters, lr=learning_rate)

    # Training Loop
    print("Training RWKV model...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Epoch {epoch + 1}/{epochs}")
        state = None

        for batch_idx, (input_ids, target_ids) in progress_bar:
            optimizer.zero_grad()
            outputs, state = model(input_ids, state)  # model expects inputs as batches

            # Detach state to avoid gradient computation across batches
            state = [s.detach() for s in state]

            # Compute loss and backpropagate
            loss = F.cross_entropy(outputs.transpose(1, 2), target_ids)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {total_loss / len(data_loader):.4f}")

        if checkpoint_dir:
            checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), checkpoint_file)
            print(f"Saved checkpoint at {checkpoint_file}")

    torch.save(model.state_dict(), model_path)
    print("Training complete. Model saved at:", model_path)


if __name__ == "__main__":
    # TODO: Change the default file
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Directory for training data")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory to save trained model")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory to save checkpoints")
    # parser.add_argument("--model_type", type=str, default="RWKV-4-PilePlus-169M-20230520-done-ctx4096.pth", help="Type of RWKV model")
    parser.add_argument("--model_type", type=str, default="RWKV-x070-Pile-168M-20241120-ctx4096.pth", help="Type of RWKV model")
    parser.add_argument("--strategy", type=str, default="cpu fp32", help="Strategy to use when creating the model")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--vocab_size", type=int, default=50257, help="Vocabulary size")
    parser.add_argument("--max_seq_length", type=int, default=256, help="Maximum sequence length for input")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning Rate")
    parser.add_argument("--load_model", type=str, help="path to model to load for continued training")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of RWKV layers")
    parser.add_argument("--embed_dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--head_dim", type=int, default=32, help="Head dimension")
    args = parser.parse_args()

    train(args.data_dir, args.model_dir, args.checkpoint_dir, args.model_type, args.strategy, args.epochs, args.batch_size, args.vocab_size, args.max_seq_length, args.learning_rate, args.load_model, args.num_layers, args.embed_dim, args.hidden_dim, args.head_dim)