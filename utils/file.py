import random
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

normalize = transforms.Normalize(
    mean=[0.4914, 0.4822, 0.4465],
    std=[0.2023, 0.1994, 0.2010]
)
transform_default = transforms.Compose([
    transforms.ToTensor(),
    normalize
])


def default_loader(filename, transform=None):
    img_pil = Image.open(filename)
    if transform:
        img_tensor = transform(img_pil)
    else:
        img_tensor = transform_default(img_pil)
    return img_tensor


def read_images(path):
    list = []
    files = os.listdir(path)
    files.sort(key=lambda x: int(x[:-4]))  # 倒着数第四位'.'为分界线，按照‘.’左边的数字从小到大排序
    for index, item in enumerate(files):
        list.append((item, index))
    return list


class ReadSet:
    def __init__(self, filename, image_dir, count=None, shuffle=True, transform=None):
        """
        :param filename: 文件名
        :param count: 切分训练集和测试集的数，count为训练集的样本数，默认50000
        :param shuffle: 是否洗牌
        """
        self.filename = filename
        self.image_dir = image_dir
        self.count = count
        self.shuffle = shuffle
        self.transform = transform
        self.result = [[] for _ in range(10)]
        self.train_list = []
        self.test_list = []
        self.train_set = []
        self.test_set = []
        # 数据读取
        self.read_list()

    def read_list(self):
        """
        当有count时，切分后分别读取到train_list和test_list中；
        当无count时，全部读取到train_list中
        :return:
        """
        with open(self.filename, "r") as f:
            for line in f.readlines():
                content = line.strip('\n').split(' ')  # 去掉列表中每一个元素的换行符，然后分割
                name = content[0]
                labels = int(content[1])
                self.result[labels].append((name, labels))
        # 洗牌
        if self.shuffle:
            # 组内乱序
            for i in range(0, len(self.result)):
                random.shuffle(self.result[i])
            # 组间乱序
            random.shuffle(self.result)
        # 切分
        if self.count:
            length = len(self.result)
            point = int(self.count / length)  # 切分点为总数/标签数=每个标签下切分的数量
            for i in range(0, len(self.result)):
                self.train_list += self.result[i][0: point]
                self.test_list += self.result[i][point:]
        else:
            for i in range(0, len(self.result)):
                self.train_list += self.result[i]

    def get_train_set(self):
        return TrainSet(self.train_list, self.image_dir, self.transform)

    def get_test_set(self):
        return TrainSet(self.test_list, self.image_dir, self.transform)


class TrainSet(Dataset):
    def __init__(self, data_list=None, image_dir=None, transform=None):
        """
        :param data_list: 数据集list [['0.jpg', 0], ['1.jpg', 8]]
        :param image_dir: 图片路径：image_dir+imge_name.jpg构成图片的完整路径
        :param transform: 图像增强方法
        """
        self.data_list = data_list
        self.image_dir = image_dir
        self.transform = transform

    def __getitem__(self, idx):
        image_name, image_label = self.data_list[idx]
        image_path = os.path.join(self.image_dir, image_name)
        image_tensor = default_loader(image_path, self.transform)
        image_label = np.array(image_label)
        return image_tensor, image_label

    def __len__(self):
        return len(self.data_list)


if __name__ == "__main__":
    rs = ReadSet(filename='../Datasets/CIFAR-10/clean_label.txt', image_dir='../Datasets/CIFAR-10/clean/', count=50000)
    trs = rs.get_train_set()
    tes = rs.get_test_set()
    print(len(trs), len(tes))
