import torch

from .hinge_loss import hinge_loss


def compute_discriminator_loss(fakeD, trueD, diff_person):
    # fake part
    loss_fake = 0
    for di in fake_D:
        loss_fake += torch.sum(
            hinge_loss(di[0], False).mean(dim=[1, 2, 3]) * diff_person
        ) / (diff_person.sum() + 1e-4)

    # ground truth part
    loss_true = 0
    for di in true_D:
        loss_true += torch.sum(
            hinge_loss(di[0], True).mean(dim=[1, 2, 3]) * diff_person
        ) / (diff_person.sum() + 1e-4)

    lossD = 0.5 * (loss_true.mean() + loss_fake.mean())

    return lossD
