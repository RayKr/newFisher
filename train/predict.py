import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from net import model_type, pre_model_path
from utils.file import TrainSet, read_images

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
# 准备数据集并预处理
transform_train = transforms.Compose([
    # 训练集上做数据增强
    # transforms.RandomCrop(32, padding=4),  # 先四周填充0，在把图像随机裁剪成32*32
    # transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    normalize,
])


def predict(model_name, pre_model_name, input_path, out_filename):
    """
    执行预测
    :param model_name: 训练模型名称
    :param pre_model_name: 预训练模型名称
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
    net = model_type[model_name].to(device)
    path = pre_model_path[pre_model_name]
    net.load_state_dict(torch.load(path))
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
            name = a_images[labels.item()][0].replace(input_path, '')
            print(name, predicted.item())
            f.write('%s %s' % (name, predicted.item()))
            f.write('\n')
            f.flush()


if __name__ == "__main__":
    # predict('resnet32', 'clean', '../Datasets/predict/BJPG/', '../Datasets/predict/result/clean/BJPG.txt')
    # predict('resnet32', 'fgsm', '../Datasets/predict/BJPG/', '../Datasets/predict/result/fgsm/BJPG.txt')
    # predict('resnet32', 'rfgsm', '../Datasets/predict/BJPG/', '../Datasets/predict/result/rfgsm/BJPG.txt')
    # predict('resnet32', 'pgd', '../Datasets/predict/BJPG/', '../Datasets/predict/result/pgd/BJPG.txt')
    # predict('resnet32', 'mixed', '../Datasets/predict/BJPG/', '../Datasets/predict/result/mixed/BJPG.txt')
    predict('swin-t', 'swin-t', '../Datasets/predict/upJPG/', '../Datasets/predict/result/swin-t/提高JPG.txt')
