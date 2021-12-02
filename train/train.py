import torch
import torch.nn as nn
import torch.optim as optim

from data import cl_test_loader, adv_loader, cl_train_loader
from eval import eval_acc
from net import model_type

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('current device=', device)
# Cifar-10的标签
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def train(train_set, model_name='resnet32', pre_model_path=None, lr=0.01, pre_epoch=0, epochs=100):
    # 模型定义-ResNet
    net = model_type[model_name].to(device)

    # 定义损失函数和优化方式
    # 损失函数为交叉熵，多用于多分类问题
    criterion = nn.CrossEntropyLoss()
    # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    best_acc = 85  # 2 初始化best test accuracy
    print("Start Training, Resnet!")  # 定义遍历数据集的次数
    with open("acc.txt", "w") as f:
        with open("log.txt", "w") as f2:
            # 如果pre_epoch不为0，则判断时手动调参后，继续训练，所以需要读取上次保存的模型参数
            if pre_model_path:
                net.load_state_dict(torch.load(pre_model_path))
            for epoch in range(pre_epoch, epochs):
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

                    # fgsm_attack
                    # inputs = fgsm_attack(net, device, inputs, labels, 0.1)

                    # pgd_attack
                    # inputs = rfgsm_attack(net, device, inputs, labels)

                    # 重新计算一遍loss
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)  # 计算前向loss
                    optimizer.zero_grad()
                    loss.backward()
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
                    # fgsm_acc = eval_acc(fgsm_test_loader, net, device)
                    # print('FGSM对抗样本 分类准确率为：%.3f%%' % fgsm_acc)
                    # pgd_acc = eval_acc(pgd_test_loader, net, device)
                    # print('PGD对抗样本 分类准确率为：%.3f%%' % pgd_acc)
                    # f.write("EPOCH=%03d, Accuracy=%.3f%%, Adv_Accuracy=%.3f%%, FGSM_Accuracy=%.3f%%, PGD_Accuracy=%.4f%%" % (epoch + 1, clean_acc, adv_acc, fgsm_acc, pgd_acc))
                    f.write("EPOCH=%03d, Accuracy=%.3f%%, Adv_Accuracy=%.3f%%" % (epoch + 1, clean_acc, adv_acc))
                    f.write('\n')
                    f.flush()

                    # 将每次测试结果实时写入acc.txt文件中
                    print('Saving model......')
                    torch.save(net.state_dict(), './tmp/net_%03d.pth' % (epoch + 1))

                    # 记录最佳测试分类准确率并写入best_acc.txt文件中
                    if clean_acc > best_acc:
                        f3 = open("best_acc.txt", "w")
                        f3.write("EPOCH=%d,best_acc= %.3f%%" % (epoch + 1, clean_acc))
                        f3.close()
                        best_acc = clean_acc
            print("Training Finished, TotalEPOCH=%d" % epochs)


# 训练
if __name__ == "__main__":
    # train(mixed_loader, pre_model_path='./net/transfer_clean_adv/net_040.pth')
    # 1.先用100%clean数据训预训练模型
    # 0.01|40 -> 0.001|60 -> 0.0001|70
    train(cl_train_loader, pre_model_path=None, lr=0.01, pre_epoch=0, epochs=100)
    # 2.rfgsm对抗训练
    # 使用预训练模型，0.01|104 -> 0.001|125
    # train(cl_train_loader, pre_model_path='./tmp/net_160.pth', lr=0.00001, pre_epoch=160, epochs=200)
    # 3.学习Adv对抗样本
    # train(mixed_loader, pre_model_path='./net/pre_rfgsm/net_125_best.pth', lr=0.01, pre_epoch=0, epochs=200)

    # Swin-T
    # 0.01 | 16 -> 0.001 | 200
    # train(cl_train_loader, model_name='swin-t', pre_model_path='./tmp/net_208.pth', lr=0.0001, pre_epoch=208, epochs=300)

    # ResNet32
    # train(mixed_loader, pre_model_path='./tmp/net_100.pth', lr=0.0001, pre_epoch=100, epochs=200)
