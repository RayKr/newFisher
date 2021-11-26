import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim

from data import cl_test_loader, adv_loader, fgsm_test_loader, pgd_test_loader, cl_train_loader, mixed_loader
from eval import eval_acc
from model.ResNet import resnet32_cifar

# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('current device=', device)

# 参数设置,使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--outf', default='./tmp/', help='folder to output images and model checkpoints')  # 输出结果保存路径
args = parser.parse_args()

# 超参数设置
EPOCH = 160  # 遍历数据集次数
pre_epoch = 0  # 定义已经遍历数据集的次数
LR = 0.001  # 学习率

# Cifar-10的标签
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 模型定义-ResNet
net = resnet32_cifar().to(device)

# 定义损失函数和优化方式
# 损失函数为交叉熵，多用于多分类问题
criterion = nn.CrossEntropyLoss()
# 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)


def train(train_set, pre_model_path=None):
    if not os.path.exists(args.outf):
        os.makedirs(args.outf)
    best_acc = 85  # 2 初始化best test accuracy
    print("Start Training, Resnet!")  # 定义遍历数据集的次数
    with open("acc.txt", "w") as f:
        with open("log.txt", "w") as f2:
            # 如果pre_epoch不为0，则判断时手动调参后，继续训练，所以需要读取上次保存的模型参数
            if pre_model_path:
                net.load_state_dict(torch.load(pre_model_path))
            for epoch in range(pre_epoch, EPOCH):
                print('\nEpoch: %d' % (epoch + 1))
                net.train()
                sum_loss = 0.0
                correct = 0.0
                total = 0.0
                for i, data in enumerate(train_set, 0):
                    # 准备数据
                    length = len(train_set)
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
                    loss.backward()  # 反向传播计算梯度

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


# 训练
if __name__ == "__main__":
    train(mixed_loader, pre_model_path='./net/transfer_clean_adv/net_040.pth')
