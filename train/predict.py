import torch
from torch.utils.data import DataLoader

from model.ResNet import resnet20_cifar
from utils.file import TrainSet, read_images

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 获取数据
a_images = read_images('../Datasets/CIFAR-10/A/')
a_data = TrainSet(data_list=a_images, image_dir='../Datasets/CIFAR-10/A/')
a_loader = DataLoader(a_data, batch_size=1, shuffle=False, num_workers=2)


if __name__ == "__main__":
    # 加载模型
    net = resnet20_cifar().to(device)
    net.load_state_dict(torch.load('./net/net_clean_136.pth'))
    net.eval()

    # 加载数据
    file_list = []
    with open("../Datasets/CIFAR-10/predict.txt", "w") as f:
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

