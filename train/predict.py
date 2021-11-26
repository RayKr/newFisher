import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from model.ResNet import resnet20_cifar, resnet32_cifar
from utils.file import TrainSet, read_images

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
# 准备数据集并预处理
transform_train = transforms.Compose([
    # 训练集上做数据增强
    transforms.RandomCrop(32, padding=4),  # 先四周填充0，在把图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    normalize,
])


def predict(model_path, input_path, out_filename):
    """
    执行预测
    :param model_path: 训练模型路径
    :param input_path: 输入的预测集路径
    :param out_filename: 输出的结果的文件名
    :return:
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 获取数据
    a_images = read_images(input_path)
    a_data = TrainSet(a_images, transform=transform_train)
    a_loader = DataLoader(a_data, batch_size=1, shuffle=False, num_workers=2)

    # 加载模型
    net = resnet32_cifar().to(device)
    net.load_state_dict(torch.load(model_path))
    net.eval()

    # 加载数据
    file_list = []
    with open(out_filename, "w") as f:
        for inputs, labels in a_loader:
            inputs = inputs.to(device)
            outputs = net(inputs)
            # 取得分最高的那个类 (outputs.data的索引号)
            _, predicted = torch.max(outputs.data, 1)
            # 重新组织list内容
            file_list.append((a_images[labels.item()][0], predicted.item()))
            print(a_images[labels.item()][0], predicted.item())
            f.write('%s %s' % (a_images[labels.item()][0], predicted.item()))
            f.write('\n')
            f.flush()


if __name__ == "__main__":
    # clean+jpg的训练模型
    # predict('./net/old/net_clean_136_jpg_best.pth', '../Datasets/predict/A/', '../Datasets/predict/result/clean_jpg/A.txt')
    # predict('./net/old/net_clean_136_jpg_best.pth', '../Datasets/predict/B/', '../Datasets/predict/result/clean_jpg/B.txt')
    # predict('./net/old/net_clean_136_jpg_best.pth', '../Datasets/predict/clean/', '../Datasets/predict/result/clean_jpg/clean.txt')

    # clean+png的训练模型
    # predict('./net/transfer_clean_adv/net_059.pth', '../Datasets/predict/A/', '../Datasets/predict/result/transfer/A.txt')
    # predict('./net/transfer_clean_adv/net_059.pth', '../Datasets/predict/B/', '../Datasets/predict/result/transfer/B.txt')
    # predict('./net/transfer_clean_adv/net_059.pth', '../Datasets/predict/clean/', '../Datasets/predict/result/transfer/clean.txt')
    predict('./net/transfer_clean_adv/net_059.pth', '../Datasets/predict/up/', '../Datasets/predict/result/transfer/提高.txt')
