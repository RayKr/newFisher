import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os

from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from attack.fgm import FGM
from attack.fgsm import fgsm_attack
from attack.pgd import pgd_attack
from model.ResNet import resnet20_cifar, resnet32_cifar
from eval import eval_acc
from utils.file import TrainSet, read_list

# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('current device=', device)

# 参数设置,使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--outf', default='./net/', help='folder to output images and model checkpoints')  # 输出结果保存路径
args = parser.parse_args()

# 超参数设置
EPOCH = 200  # 遍历数据集次数
pre_epoch = 156  # 定义已经遍历数据集的次数
BATCH_SIZE = 120  # 批处理尺寸(batch_size)
LR = 0.001  # 学习率

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
transform_test = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])

# 读取数据
cltrain_list, cltest_list = read_list('../Datasets/CIFAR-10/clean_label.txt', 50000)
cltrain_data = TrainSet(data_list=cltrain_list, image_dir='../Datasets/CIFAR-10/clean/', transform=transform_train)
cltest_data = TrainSet(data_list=cltest_list, image_dir='../Datasets/CIFAR-10/clean/', transform=transform_test)
cltrain_loader = DataLoader(cltrain_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
cltest_loader = DataLoader(cltest_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# 对抗样本训练集和测试集
advtrain_list, advtest_list = read_list('../Datasets/CIFAR-10/clean_label.txt', 50000)
advtrain_data = TrainSet(data_list=advtrain_list, image_dir='../Datasets/gen_adv/fgsm_0.15/', transform=transform_train)
advtest_data = TrainSet(data_list=advtest_list, image_dir='../Datasets/gen_adv/fgsm_0.15/', transform=transform_test)
advtrain_loader = DataLoader(advtrain_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
advtest_loader = DataLoader(advtest_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# 对抗样本验证集
adv_list = read_list('../Datasets/CIFAR-10/adv.txt')
adv_data = TrainSet(data_list=adv_list, image_dir='../Datasets/CIFAR-10/adv/')
adv_loader = DataLoader(adv_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# Cifar-10的标签
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 模型定义-ResNet
net = resnet20_cifar().to(device)

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
    print("Start Training, Resnet-18!")  # 定义遍历数据集的次数
    with open("acc.txt", "w") as f:
        with open("log.txt", "w") as f2:
            # 如果pre_epoch不为0，则判断时手动调参后，继续训练，所以需要读取上次保存的模型参数
            if pre_epoch != 0:
                net.load_state_dict(torch.load('./net/net_%03d.pth' % pre_epoch))
            for epoch in range(pre_epoch, EPOCH):
                print('\nEpoch: %d' % (epoch + 1))
                net.train()
                sum_loss = 0.0
                correct = 0.0
                total = 0.0
                for i, data in enumerate(cltrain_loader, 0):
                    # 准备数据
                    length = len(cltrain_loader)
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)

                    # 加FGSM攻击
                    inputs = fgsm_attack(net, device, inputs, labels, 0.15)

                    # pdg攻击
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
                    clean_acc = eval_acc(cltest_loader, net, device)
                    print('测试Clean分类准确率为：%.3f%%' % clean_acc)
                    adv_acc = eval_acc(adv_loader, net, device)
                    print('测试Adv分类准确率为：%.3f%%' % adv_acc)
                    f.write("EPOCH=%03d, Accuracy=%.3f%%, Adv_Accuracy=%.3f%%" % (epoch + 1, clean_acc, adv_acc))
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
