import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim, 2),
        )

    def forward(self, x):
        return self.classifier(x)
