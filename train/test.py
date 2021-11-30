import torch

from attack.fgsm import fgsm_attack
from model.ResNet import resnet20_cifar, resnet32_cifar
from eval import eval_acc
from data import cl_test_loader, adv_loader, fgsm_test_loader, pgd_test_loader
from model.swin_transformer import SwinTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 加载模型
# net = resnet32_cifar().to(device)
net = SwinTransformer().to(device)


def attack_test(filename, desc=None, attack_method=None, **kwargs):
    net.load_state_dict(torch.load(filename))
    net.eval()

    if desc is not None:
        print(desc)
    clean_acc = eval_acc(cl_test_loader, net, device, attack_method=attack_method, **kwargs)
    adv_acc = eval_acc(adv_loader, net, device, attack_method=attack_method, **kwargs)
    print('Clean_Acc = %.3f%%, Adv_Acc = %.3f%%' % (clean_acc, adv_acc))
    # fgsm_acc = eval_attack_acc(fgsm_test_loader, net, device, attack_method=attack_method, **kwargs)
    # print('Generated FGSM Adversarial Examples Acc = %.3f%%' % fgsm_acc)
    # pgd_acc = eval_attack_acc(pgd_test_loader, net, device, attack_method=attack_method, **kwargs)
    # print('Generated PGD Adversarial Examples Acc = %.3f%%' % pgd_acc)


def robust_test(model_path, model_title):
    print(f'--------------{model_title}--------------')
    attack_test(model_path, '【无攻击】')
    attack_test(model_path, '【FGSM攻击：epsilon=0.1】', attack_method='fgsm', epsilon=0.1)
    attack_test(model_path, '【PGD攻击：epsilon=0.3, alpha=2/255, iters=20】', attack_method='pgd', epsilon=0.3, alpha=2/255, iters=20)
    attack_test(model_path, '【RFGSM攻击：alpha=0.1, eps=0.5】', attack_method='rfgsm', alpha=0.1, eps=0.5)
    attack_test(model_path, '【FGM攻击：epsilon=0.15】', attack_method='fgm', epsilon=0.15)


if __name__ == "__main__":
    # robust_test('./net/pre_train/net_070.pth', 'Clean预训练模型')
    # robust_test('./net/pgd/net_150.pth', 'PGD对抗训练模型')
    robust_test('./net/swin-t/net_010.pth', 'Swin-T')
