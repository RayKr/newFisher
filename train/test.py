import torch

from attack.fgsm import fgsm_attack
from model.ResNet import resnet20_cifar, resnet32_cifar
from eval import eval_acc
from data import cl_test_loader, adv_loader, fgsm_test_loader, pgd_test_loader
from eval_attack import eval_attack_acc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 加载模型
net = resnet32_cifar().to(device)


def do_test(filename, desc=None):
    net.load_state_dict(torch.load(filename))
    net.eval()

    if desc is not None:
        print(desc)
    clean_acc = eval_acc(cl_test_loader, net, device)
    print('Clean Examples Acc = %.3f%%' % clean_acc)
    adv_acc = eval_acc(adv_loader, net, device)
    print('Adv Examples Acc = %.3f%%' % adv_acc)
    fgsm_acc = eval_acc(fgsm_test_loader, net, device)
    print('Generated FGSM Adversarial Examples Acc = %.3f%%' % fgsm_acc)
    pgd_acc = eval_acc(pgd_test_loader, net, device)
    print('Generated PGD Adversarial Examples Acc = %.3f%%' % pgd_acc)


def attack_test(filename, desc=None, attack_method=None, **kwargs):
    net.load_state_dict(torch.load(filename))
    net.eval()

    if desc is not None:
        print(desc)
    clean_acc = eval_attack_acc(cl_test_loader, net, device, attack_method=attack_method, **kwargs)
    print('Clean Examples Acc = %.3f%%' % clean_acc)
    adv_acc = eval_attack_acc(adv_loader, net, device, attack_method=attack_method, **kwargs)
    print('Adv Examples Acc = %.3f%%' % adv_acc)
    fgsm_acc = eval_attack_acc(fgsm_test_loader, net, device, attack_method=attack_method, **kwargs)
    print('Generated FGSM Adversarial Examples Acc = %.3f%%' % fgsm_acc)
    pgd_acc = eval_attack_acc(pgd_test_loader, net, device, attack_method=attack_method, **kwargs)
    print('Generated PGD Adversarial Examples Acc = %.3f%%' % pgd_acc)


if __name__ == "__main__":
    # do_test('./net/net_CIFAR_Down_clean_160.pth', '这是用CIFAR下载的数据包训练的模型')
    # do_test('./net/net_new_clean_160.pth', '这是用clean文件夹训练的模型')
    # do_test('./net/net_clean_fgsm_082.pth', '这是用clean+fgsm训练的模型')
    # do_test('./net/silu/net_clean_silu_72.pth', '这是用silu训练的模型')
    # do_test('net/old/net_clean_136_jpg_best.pth', '这是用jpg训练的模型')
    # do_test('./net/net_new_clean_160.pth', '这是用新数据集png训练的模型')
    # do_test('./net/mixed_clean_adv/net_131.pth', '这是用干净样本训练的模型')
    # do_test('./net/transfer_clean_adv/net_040.pth', '这是用迁移训练的模型')
    # do_test('./net/clean/net_080.pth', '这是用clean样本训练的模型')
    # do_test('./net/fgsm/net_177.pth', '这是用fgsm对抗训练的模型')
    # do_test('./net/pgd/net_150.pth', '这是用pgd对抗训练的模型')
    # attack_test('./net/clean/net_080.pth', desc='clean模型，fgsm攻击', attack_method='fgsm', epsilon=0.1)
    # attack_test('./net/fgsm/net_177.pth', desc='fgsm模型，fgsm攻击', attack_method='fgsm', epsilon=0.1)
    # attack_test('./net/pgd/net_150.pth', desc='pgd模型，fgsm攻击', attack_method='fgsm', epsilon=0.1)
    # attack_test('./net/clean/net_080.pth', desc='clean模型，pgd攻击', attack_method='pgd')
    # attack_test('./net/fgsm/net_177.pth', desc='fgsm模型，pgd攻击', attack_method='pgd')
    # attack_test('./net/pgd/net_150.pth', desc='pgd模型，pgd攻击', attack_method='pgd')
    attack_test('./net/transfer_clean_adv/net_040.pth', desc='高分模型，pgd攻击', attack_method='pgd')