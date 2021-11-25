import torch
import torchvision
from torch.utils.data import DataLoader

from attack.fgsm import fgsm_attack
from model.ResNet import resnet20_cifar
from eval import eval_acc
from utils.file import ReadSet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 获取数据
read_set = ReadSet(filename='../Datasets/CIFAR-10/clean_label.txt', image_dir='../Datasets/CIFAR-10/clean_png/', count=50000)
test_data = read_set.get_test_set()
test_loader = DataLoader(test_data, batch_size=100, shuffle=False, num_workers=2)

adv_set = ReadSet(filename='../Datasets/CIFAR-10/adv.txt', image_dir='../Datasets/CIFAR-10/adv/', shuffle=False)
adv_data = adv_set.get_train_set()
adv_loader = DataLoader(adv_data, batch_size=120, shuffle=False, num_workers=2)


def do_test(filename, desc=None):
    # 加载模型
    net = resnet20_cifar().to(device)
    net.load_state_dict(torch.load(filename))
    net.eval()

    if desc is not None:
        print(desc)
    clean_acc = eval_acc(test_loader, net, device)
    # clean_acc = eval_acc(test_loader, net, device, fgsm_attack, 0.03)
    print('Adversarial Attack: FGSM, epsilon=0.2')
    print('Clean Examples Acc = %.3f%%' % clean_acc)
    adt_acc = eval_acc(adt_loader, net, device)
    print('Generated FGSM Adversarial Examples Acc = %.3f%%' % adt_acc)
    adv_acc = eval_acc(adv_loader, net, device)
    print('Adv Examples Acc = %.3f%%' % adv_acc)


if __name__ == "__main__":
    # do_test('./net/net_CIFAR_Down_clean_160.pth', '这是用CIFAR下载的数据包训练的模型')
    # do_test('./net/net_new_clean_160.pth', '这是用clean文件夹训练的模型')
    # do_test('./net/net_clean_fgsm_082.pth', '这是用clean+fgsm训练的模型')
    # do_test('./net/silu/net_clean_silu_72.pth', '这是用silu训练的模型')
    # do_test('net/old/net_clean_136_jpg_best.pth', '这是用jpg训练的模型')
    do_test('net/net_new_clean_160.pth', '这是用新数据集png训练的模型')
