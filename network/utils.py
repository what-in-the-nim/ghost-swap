from torch import nn


def weight_init(module: nn.Module) -> None:
    """Initialize weights for the torch module."""
    # Initialize weights for linear layers
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(0, 0.001)
        module.bias.data.zero_()
    # Initialize weights for convolutional layers
    if isinstance(module, nn.Conv2d):
        nn.init.xavier_normal_(module.weight.data)
    # Initialize weights for transpose convolutional layers
    if isinstance(module, nn.ConvTranspose2d):
        nn.init.xavier_normal_(module.weight.data)
