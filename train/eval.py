import torch
from torchvision import transforms

toPIL = transforms.ToPILImage()


def eval_acc(dataloader, net, device, attack=None, epsilon=0.03):
    total, correct = 0, 0
    for data in dataloader:
        net.eval()
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        # 加攻击
        if attack:
            images = attack(net, device, images, labels, epsilon)

        outputs = net(images)
        # 取得分最高的那个类 (outputs.data的索引号)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
        # print('当前数：%d, 正确数：%d, Acc=%.3f%%' % (total, correct, 100. * correct / total))
    acc = 100. * correct / total
    return acc
