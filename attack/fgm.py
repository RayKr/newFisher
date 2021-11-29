import torch
from torch import nn


def fgm_attack(model, device, inputs, labels, epsilon=0.15):
    criterion = nn.CrossEntropyLoss()
    inputs = inputs.to(device)
    labels = labels.to(device)
    # 正常训练
    inputs.requires_grad = True
    outputs = model(inputs)
    loss = criterion(outputs, labels)  # 计算前向loss
    model.zero_grad()
    loss.backward()  # 反向传播计算梯度

    # 加FGM攻击
    data_grad = inputs.grad.data
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = inputs + epsilon * data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image
