import torch


def fgsm_attack(model, device, criterion, inputs, labels, epsilon):
    inputs = inputs.to(device)
    labels = labels.to(device)
    # 正常训练
    inputs.requires_grad = True
    outputs = model(inputs)
    loss = criterion(outputs, labels)  # 计算前向loss
    model.zero_grad()
    loss.backward()  # 反向传播计算梯度

    # 加FGSM攻击
    data_grad = inputs.grad.data
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = inputs + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image
