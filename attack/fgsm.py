import torch
from torch import nn


def fgsm_attack(model, device, images, labels, eps=0.3):
    images = images.to(device)
    labels = labels.to(device)
    loss = nn.CrossEntropyLoss()
    images.requires_grad = True

    outputs = model(images)
    model.zero_grad()
    cost = loss(outputs, labels).to(device)
    cost.backward()

    # 图像 + 梯度得到对抗样本
    grad = images.grad.data
    # 使用sign（符号）函数，将对x求了偏导的梯度进行符号化
    adv_images = images + eps * grad.sign()
    # 做一个剪裁的工作，将torch.clamp内部大于1的数值变为1，小于0的数值等于0，防止image越界
    adv_images = torch.clamp(adv_images, 0, 1)

    return adv_images

