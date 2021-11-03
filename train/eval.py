import torch


def eval_acc(dataloader, net, device):
    total, correct = 0, 0
    for data in dataloader:
        net.eval()
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        # 取得分最高的那个类 (outputs.data的索引号)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    acc = 100. * correct / total
    return acc
