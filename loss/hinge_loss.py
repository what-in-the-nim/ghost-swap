import torch


def hinge_loss(X: torch.Tensor, positive: bool = True) -> torch.Tensor:
    """Hinge loss function"""
    if positive:
        return torch.relu(1 - X)
    else:
        return torch.relu(X + 1)
