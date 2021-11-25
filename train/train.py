import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from attack.fgsm import fgsm_attack
from attack.pgd import pgd_attack
from eval import eval_acc
from model.ResNet import resnet20_cifar, preact_resnet1001_cifar, resnet32_cifar
from utils.file import ReadSet

# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('current device=', device)

# 参数设置,使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--outf', default='./tmp/', help='folder to output images and model checkpoints')  # 输出结果保存路径
args = parser.parse_args()

# 超参数设置
conf = [(10, 0.01), (80, 0.001)]
EPOCH = 160  # 遍历数据集次数
pre_epoch = 0  # 定义已经遍历数据集的次数
BATCH_SIZE = 128  # 批处理尺寸(batch_size)
LR = 0.01  # 学习率

normalize = transforms.Normalize(
    mean=[0.4914, 0.4822, 0.4465],
    std=[0.2023, 0.1994, 0.2010]
)
# 准备数据集并预处理
transform_train = transforms.Compose([
    # 训练集上做数据增强
    transforms.RandomCrop(32, padding=4),  # 先四周填充0，在把图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    normalize,
])

# 读取数据
# cifar_train = torchvision.datasets.CIFAR10(root='../Datasets/', train=True, download=True, transform=transform_train)
# cifar_test = torchvision.datasets.CIFAR10(root='../Datasets/', train=False, download=True, transform=transform_train)
# cl_train_loader = DataLoader(cifar_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
# cl_test_loader = DataLoader(cifar_test, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
read_set = ReadSet(filename='../Datasets/CIFAR-10/clean_label.txt', image_dir='../Datasets/CIFAR-10/clean_png/', count=50000, transform=transform_train, dct=True)
cl_train_data = read_set.get_train_set()
cl_test_data = read_set.get_test_set()
cl_train_loader = DataLoader(cl_train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
cl_test_loader = DataLoader(cl_test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# pdg生成对抗样本
pgd_set = ReadSet(filename='../Datasets/CIFAR-10/clean_label.txt', image_dir='../Datasets/gen_adv/pgd/', count=50000)
pgd_train_data = pgd_set.get_train_set()
pgd_test_data = pgd_set.get_test_set()
pgd_train_loader = DataLoader(pgd_train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
pgd_test_loader = DataLoader(pgd_test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# fgsm生成的对抗样本
fgsm_set = ReadSet(filename='../Datasets/CIFAR-10/clean_label.txt', image_dir='../Datasets/gen_adv/fgsm_0.1/', count=50000)
fgsm_train_data = fgsm_set.get_train_set()
fgsm_test_data = fgsm_set.get_test_set()
fgsm_train_loader = DataLoader(fgsm_train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
fgsm_test_loader = DataLoader(fgsm_test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# 对抗样本验证集
adv_set = ReadSet(filename='../Datasets/CIFAR-10/adv.txt', image_dir='../Datasets/CIFAR-10/adv/', shuffle=False, transform=transform_train)
adv_data = adv_set.get_train_set()
adv_loader = DataLoader(adv_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# Cifar-10的标签
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 模型定义-ResNet
net = resnet32_cifar().to(device)

# 定义损失函数和优化方式
# 损失函数为交叉熵，多用于多分类问题
criterion = nn.CrossEntropyLoss()
# 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)

# 训练
if __name__ == "__main__":
    if not os.path.exists(args.outf):
        os.makedirs(args.outf)
    best_acc = 85  # 2 初始化best test accuracy
    print("Start Training, Resnet!")  # 定义遍历数据集的次数
    with open("acc.txt", "w") as f:
        with open("log.txt", "w") as f2:
            # 如果pre_epoch不为0，则判断时手动调参后，继续训练，所以需要读取上次保存的模型参数
            if pre_epoch != 0:
                net.load_state_dict(torch.load('./tmp/net_%03d.pth' % pre_epoch))
            for epoch in range(pre_epoch, EPOCH):
                print('\nEpoch: %d' % (epoch + 1))
                net.train()
                sum_loss = 0.0
                correct = 0.0
                total = 0.0
                for i, data in enumerate(cl_train_loader, 0):
                    # 准备数据
                    length = len(cl_train_loader)
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)

                    # 加FGSM攻击
                    # inputs = fgsm_attack(net, device, inputs, labels, 0.03)

                    # pgd攻击
                    # inputs = pgd_attack(net, device, inputs, labels)

                    optimizer.zero_grad()

                    # 正常训练
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)  # 计算前向loss
                    loss.backward()  # 反向传播计算梯度，注意先不要更新梯度，即optimizer.step()

                    # 梯度下降
                    optimizer.step()

                    # 每训练1个batch打印一次loss和准确率
                    sum_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += predicted.eq(labels.data).cpu().sum()
                    print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% '
                             % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('\n')
                    f2.flush()

                # 每训练完一个epoch测试一下准确率
                print("Waiting Test!")
                with torch.no_grad():
                    clean_acc = eval_acc(cl_test_loader, net, device)
                    print('Clean测试集 分类准确率为：%.3f%%' % clean_acc)
                    adv_acc = eval_acc(adv_loader, net, device)
                    print('Adv测试集 分类准确率为：%.3f%%' % adv_acc)
                    fgsm_acc = eval_acc(fgsm_test_loader, net, device)
                    print('FGSM对抗样本 分类准确率为：%.3f%%' % fgsm_acc)
                    pgd_acc = eval_acc(pgd_test_loader, net, device)
                    print('PGD对抗样本 分类准确率为：%.3f%%' % pgd_acc)
                    f.write("EPOCH=%03d, Accuracy=%.3f%%, Adv_Accuracy=%.3f%%, FGSM_Accuracy=%.3f%%, PGD_Accuracy=%.4f%%" % (epoch + 1, clean_acc, adv_acc, fgsm_acc, pgd_acc))
                    f.write('\n')
                    f.flush()

                    # 将每次测试结果实时写入acc.txt文件中
                    print('Saving model......')
                    torch.save(net.state_dict(), '%s/net_%03d.pth' % (args.outf, epoch + 1))

                    # 记录最佳测试分类准确率并写入best_acc.txt文件中
                    if clean_acc > best_acc:
                        f3 = open("best_acc.txt", "w")
                        f3.write("EPOCH=%d,best_acc= %.3f%%" % (epoch + 1, clean_acc))
                        f3.close()
                        best_acc = clean_acc
            print("Training Finished, TotalEPOCH=%d" % EPOCH)
