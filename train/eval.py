import torch

from attack.fgm import fgm_attack
from attack.fgsm import fgsm_attack
from attack.pgd import pgd_attack
from attack.rfgsm import rfgsm_attack

attack_list = {'fgsm': fgsm_attack, 'pgd': pgd_attack, 'rfgsm': rfgsm_attack, 'fgm': fgm_attack}


def eval_acc(dataloader, net, device, attack_method=None, **kwargs):
    total, correct = 0, 0
    for data in dataloader:
        net.eval()
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        if attack_method:
            att = attack_list[attack_method]
            images = att(net, device, images, labels, **kwargs)

        outputs = net(images)
        # 取得分最高的那个类 (outputs.data的索引号)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
        # print('当前数：%d, 正确数：%d, Acc=%.3f%%' % (total, correct, 100. * correct / total))
    acc = 100. * correct / total
    return acc
