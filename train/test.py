import torch
from torch.utils.data import DataLoader

from attack.fgsm import fgsm_attack
from model.ResNet import resnet20_cifar
from eval import eval_acc
from utils.file import read_list, TrainSet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 获取数据
_, test_list = read_list('../Datasets/CIFAR-10/clean_label.txt', 50000)
test_data = TrainSet(data_list=test_list, image_dir='../Datasets/CIFAR-10/clean/')
test_loader = DataLoader(test_data, batch_size=120, shuffle=False, num_workers=2)

adv_list = read_list('../Datasets/CIFAR-10/adv.txt')
adv_data = TrainSet(data_list=adv_list, image_dir='../Datasets/CIFAR-10/adv/')
adv_loader = DataLoader(adv_data, batch_size=120, shuffle=False, num_workers=2)


def do_test(epoch=1, desc=None):
    # 加载模型
    net = resnet20_cifar().to(device)
    net.load_state_dict(torch.load('./net/net_%03d.pth' % epoch))
    net.eval()

    print('----------------- Epoch=%3d -----------------' % epoch)
    if desc is not None:
        print(desc)
    clean_acc = eval_acc(test_loader, net, device, fgsm_attack)
    print('干净样本的预测正确率为：%.3f%%' % clean_acc)
    adv_acc = eval_acc(adv_loader, net, device)
    print('对抗样本的预测正确率为：%.3f%%' % adv_acc)
    print('---------------------------------------------')


if __name__ == "__main__":
    # do_test(135, 'clean测试集上的模型直接预测对抗样本（未进行对抗训练）')
    do_test(145, '对抗样本上的正确率（对抗训练后）')
    # do_test(135, 'clean和adv进行简单混合，clean是总样本的50%（未进行对抗训练）')
