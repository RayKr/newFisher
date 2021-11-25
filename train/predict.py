import torch
from torch.utils.data import DataLoader

from model.ResNet import resnet20_cifar
from utils.file import TrainSet, read_images


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
    a_data = TrainSet(a_images)
    a_loader = DataLoader(a_data, batch_size=1, shuffle=False, num_workers=2)

    # 加载模型
    net = resnet20_cifar().to(device)
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
    # predict('./net/net_new_clean_160.pth', '../Datasets/predict/A/', '../Datasets/predict/result/clean_png/A.txt')
    # predict('./net/net_new_clean_160.pth', '../Datasets/predict/B/', '../Datasets/predict/result/clean_png/B.txt')
    predict('./net/net_new_clean_160.pth', '../Datasets/predict/clean/', '../Datasets/predict/result/clean_png/clean.txt')
