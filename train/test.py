import torch

from attack.fgsm import fgsm_attack
from model.ResNet import resnet20_cifar, resnet32_cifar
from eval import eval_acc
from data import cl_test_loader, adv_loader, fgsm_test_loader, pgd_test_loader


def do_test(filename, desc=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载模型
    net = resnet32_cifar().to(device)
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


if __name__ == "__main__":
    # do_test('./net/net_CIFAR_Down_clean_160.pth', '这是用CIFAR下载的数据包训练的模型')
    # do_test('./net/net_new_clean_160.pth', '这是用clean文件夹训练的模型')
    # do_test('./net/net_clean_fgsm_082.pth', '这是用clean+fgsm训练的模型')
    # do_test('./net/silu/net_clean_silu_72.pth', '这是用silu训练的模型')
    # do_test('net/old/net_clean_136_jpg_best.pth', '这是用jpg训练的模型')
    # do_test('./net/net_new_clean_160.pth', '这是用新数据集png训练的模型')
    # do_test('./net/mixed_clean_adv/net_131.pth', '这是用干净样本训练的模型')
    do_test('./net/transfer_clean_adv/net_040.pth', '这是用迁移训练的模型')
