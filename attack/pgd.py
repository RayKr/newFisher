import torch


def pgd_attack(model, device, criterion, inputs, labels, epsilon=0.3, alpha=2/255, iters=20):
    inputs = inputs.to(device)
    labels = labels.to(device)
    # 原图像
    ori_images = inputs.data

    for i in range(iters):
        inputs.requires_grad = True
        outputs = model(inputs)
        cost = criterion(outputs, labels).to(device)
        model.zero_grad()
        cost.backward()
        # 图像 + 梯度得到对抗样本
        adv_images = inputs + alpha * inputs.grad.sign()
        # 限制扰动范围
        eta = torch.clamp(adv_images - ori_images, min=-epsilon, max=epsilon)
        # 进行下一轮对抗样本的生成。破坏之前的计算图
        inputs = torch.clamp(ori_images + eta, min=0, max=1).detach_()

    return inputs
