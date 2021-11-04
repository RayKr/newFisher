import torch
from torch import nn


def pgd_attack(model, device, images, labels, eps=0.1, alpha=0.01, iters=15):
    images = images.to(device)
    labels = labels.to(device)
    loss = nn.CrossEntropyLoss()
    # 原图像
    ori_images = images.data

    for i in range(iters):
        images.requires_grad = True
        outputs = model(images)

        model.zero_grad()
        cost = loss(outputs, labels).to(device)
        cost.backward()
        # 图像 + 梯度得到对抗样本
        adv_images = images + alpha*images.grad.sign()
        # 限制扰动范围
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        # 进行下一轮对抗样本的生成。破坏之前的计算图
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()

    return images

