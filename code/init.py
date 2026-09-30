from types import SimpleNamespace
import torch


def init_parameters(is_training=False):
    return SimpleNamespace(
        input_dim=384,
        hidden_dim=384,
        dropout_rate=0.2,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
