import torch
from torch.utils.data import DataLoader

from attack.fgsm import fgsm_attack
from model.ResNet import resnet20_cifar
from eval import eval_acc
from utils.file import read_list, TrainSet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 获取数据
_, test_list = read_list('../Datasets/CIFAR-10/clean_label.txt', 40000)
test_data = TrainSet(data_list=test_list, image_dir='../Datasets/CIFAR-10/clean/')
test_loader = DataLoader(test_data, batch_size=100, shuffle=False, num_workers=2)

_, adt_list = read_list('../Datasets/CIFAR-10/clean_label.txt', 50000)
adt_data = TrainSet(data_list=test_list, image_dir='../Datasets/gen_adv/fgsm_0.15/')
adt_loader = DataLoader(adt_data, batch_size=120, shuffle=False, num_workers=2)
#
adv_list = read_list('../Datasets/CIFAR-10/adv.txt')
adv_data = TrainSet(data_list=adv_list, image_dir='../Datasets/CIFAR-10/adv/')
adv_loader = DataLoader(adv_data, batch_size=120, shuffle=False, num_workers=2)


def do_test(filename, desc=None):
    # 加载模型
    net = resnet20_cifar().to(device)
    net.load_state_dict(torch.load(filename))
    net.eval()

    if desc is not None:
        print(desc)
    clean_acc = eval_acc(test_loader, net, device, fgsm_attack, 0.2)
    # print('Adversarial Attack: FGSM, epsilon=0.2')
    print('Clean Examples Acc = %.3f%%' % clean_acc)
    adt_acc = eval_acc(adt_loader, net, device)
    print('Generated FGSM Adversarial Examples Acc = %.3f%%' % adt_acc)
    adv_acc = eval_acc(adv_loader, net, device)
    print('Adv Examples Acc = %.3f%%' % adv_acc)


if __name__ == "__main__":
    # do_test(135, 'clean测试集上的模型直接预测对抗样本（未进行对抗训练）')
    do_test('./net/net_gen_150.pth')
    # do_test(135, 'clean和adv进行简单混合，clean是总样本的50%（未进行对抗训练）')
