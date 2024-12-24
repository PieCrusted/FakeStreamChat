import torch
import os
from rwkv.model import RWKV

def create_dummy_model(model_path, vocab_size, num_layers, embed_dim, hidden_dim, head_dim):
    """Creates a dummy RWKV model and saves it to the specified path."""
    config = {
            "num_layers": num_layers,
            "embed_dim": embed_dim,
            "vocab_size": vocab_size,
            "hidden_dim": hidden_dim,
            "head_dim": head_dim
    }

    dummy_model = RWKV(model='RWKV_HEAD_QK_DIM', strategy="cpu fp32")

    # Initialize model with random weights
    for name, param in dummy_model.named_parameters():
        if 'weight' in name or 'bias' in name:
            torch.nn.init.xavier_uniform_(param)


    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(dummy_model.state_dict(), model_path)
    print(f"Created and saved dummy model at {model_path}")


if __name__ == '__main__':
    model_path = 'models/dummy_rwkv_model.pth'
    create_dummy_model(model_path, 50257, 4, 128, 256, 32)