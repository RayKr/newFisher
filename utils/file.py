import random
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


# R,G,B每层的归一化用到的均值和方差
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


def default_loader(filename):
    img_pil = Image.open(filename)
    img_tensor = transform_train(img_pil)
    return img_tensor


def read_list(filename):
    """
    读取文件，返回图片list
    :param filename: 文件名
    :return:
    """
    list = []
    with open(filename, "r") as f:
        for line in f.readlines():
            content = line.strip('\n').split(' ')  # 去掉列表中每一个元素的换行符，然后分割
            name = content[0]
            labels = int(content[1])
            list.append((name, labels))
    return list


def read_labels(path, shuffle=False):
    result = []
    with open(path, "r") as f:
        for line in f.readlines():
            line = line.strip('\n')  # 去掉列表中每一个元素的换行符
            split = line.split(' ')
            result.append(split)
    if shuffle:
        random.shuffle(result)
    return result


def read_clean_list(filename, rate=0.8):
    """
    从clean_label.txt文件里获取训练集和测试集
    根据clean_label.txt的数据分布规律，可以每隔len行随机80%行作为训练集，剩下的作为测试集
    :param filename: 文件名
    :param rate: 划分训练集和测试集的比例，默认80%训练集，20%测试集
    """
    result = read_list(filename)
    # 将80%的数据作为训练集，20%数据作为测试集
    point = int(len(result) * rate)
    # 先打乱某一分类的样本的顺序
    random.shuffle(result)
    # 再划分
    train_list = result[0: point]
    test_list = result[point: int(len(result) + 1)]
    # print(len(train_list), len(test_list))
    return train_list, test_list


class TrainSet(Dataset):
    def __init__(self, data_list=None, image_dir=None):
        """
        :param data_list: 数据集list [['0.jpg', 0], ['1.jpg', 8]]
        :param image_dir: 图片路径：image_dir+imge_name.jpg构成图片的完整路径
        """
        self.data_list = data_list
        self.image_dir = image_dir

    def __getitem__(self, idx):
        image_name, image_label = self.data_list[idx]
        image_path = os.path.join(self.image_dir, image_name)
        image_tensor = default_loader(image_path)
        image_label = np.array(image_label)
        # print(image_tensor, image_label)
        return image_tensor, image_label

    def __len__(self):
        return len(self.data_list)


# if __name__ == "__main__":
#     # 获取图片的路径和label编号
#     train_list, test_list = read_clean_list('../Datasets/CIFAR-10/adv.txt')
#     print(test_list)
#     train_data = TrainSet(data_list=test_list, image_dir='../Datasets/CIFAR-10/clean/')
#     # print(train_data)
#     train_loader = DataLoader(train_data, batch_size=2, shuffle=True, num_workers=2)
    # print(train_loader)
    # for img, label in train_loader:
        # print(img, label)
    # image_name, label = train_list[0]
    # print(image_name, label)
    # print(train_list, test_list)
    # adv_list = read_adv_list('../Datasets/CIFAR-10/adv.txt')
    # print(adv_list)
    # 填入Dataloader，进而转成Tensor
