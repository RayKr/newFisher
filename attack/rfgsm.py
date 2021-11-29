import torch
from torch import nn


def rfgsm_attack(model, device, images, labels, alpha=0.1, eps=0.5):
    criterion = nn.CrossEntropyLoss()
    images = images.to(device)
    labels = labels.to(device)

    images_new = images + alpha * torch.randn_like(images).sign()
    images_new.requires_grad = True

    outputs = model(images_new)

    model.zero_grad()
    cost = criterion(outputs, labels).to(device)
    cost.backward()

    attack_images = images_new + (eps - alpha) * images_new.grad.sign()
    attack_images = torch.clamp(attack_images, 0, 1)

    return attack_images
