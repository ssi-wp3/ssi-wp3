import torch as nn
from skorch import NeuralNetClassifier


class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


def create_skorch_model(max_epochs: int,
                        batch_size: int,
                        lr: float,
                        test_size: float) -> NeuralNetClassifier:
    model = NeuralNetClassifier(
        LogisticRegression,
        max_epochs=max_epochs,
        batch_size=batch_size,
        lr=lr,
        train_split=test_size,
        criterion=nn.CrossEntropyLoss,
    )
    return model
