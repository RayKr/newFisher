import random
import os

import cv2
import numpy as np
from PIL import Image, ImageFilter
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


# 整张图 DCT 变换
def whole_img_dct(filename):
    img_u8 = cv2.imread(filename, 0)
    img_f32 = img_u8.astype(np.float)  # 数据类型转换 转换为浮点型
    img_dct = cv2.dct(img_f32)            # 进行离散余弦变换
    img_dct_log = np.log(abs(img_dct))    # 进行log处理
    img_idct = cv2.idct(img_dct)          # 进行离散余弦反变换
    return img_dct_log, img_idct


def default_loader(filename, transform=None, dct=False):
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
        list.append((path + item, index))
    return list


def read_list(filename, image_dir, shuffle=False, count=None):
    """
    当有count时，切分后分别读取到train_list和test_list中；
    当无count时，全部读取到train_list中
    :return:
    """
    result = [[] for _ in range(10)]
    train_list, test_list = [], []
    with open(filename, "r") as f:
        for line in f.readlines():
            content = line.strip('\n').split(' ')  # 去掉列表中每一个元素的换行符，然后分割
            name = image_dir + content[0]
            labels = int(content[1])
            result[labels].append((name, labels))
    # 洗牌
    if shuffle:
        # 组内乱序
        for i in range(0, len(result)):
            random.shuffle(result[i])
        # 组间乱序
        random.shuffle(result)
    # 切分
    if count:
        length = len(result)
        point = int(count / length)  # 切分点为总数/标签数=每个标签下切分的数量
        for i in range(0, len(result)):
            train_list += result[i][0: point]
            test_list += result[i][point:]
    else:
        for i in range(0, len(result)):
            train_list += result[i]

    return train_list, test_list


class ReadSet:
    def __init__(self, filename, image_dir, count=None, shuffle=True, transform=None, **kwargs):
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
        # 处理自定义参数
        self._parse_params(**kwargs)

        # 数据读取
        self.train_list, self.test_list = read_list(self.filename, self.image_dir, self.shuffle, self.count)

    def _parse_params(self, **kwargs):
        self.dct = kwargs.get("dct", False)

    def get_train_set(self):
        return TrainSet(self.train_list, self.transform, self.dct)

    def get_test_set(self):
        return TrainSet(self.test_list, self.transform, self.dct)


class TrainSet(Dataset):
    def __init__(self, data_list=None, transform=None, dct=False):
        """
        :param data_list: 数据集list [['0.jpg', 0], ['1.jpg', 8]]
        :param image_dir: 图片路径：image_dir+imge_name.jpg构成图片的完整路径
        :param transform: 图像增强方法
        """
        self.data_list = data_list
        self.transform = transform
        self.dct = dct

    def __getitem__(self, idx):
        image_path, image_label = self.data_list[idx]
        image_tensor = default_loader(image_path, self.transform, self.dct)
        image_label = np.array(image_label)
        return image_tensor, image_label

    def __len__(self):
        return len(self.data_list)


if __name__ == "__main__":
    # rs = ReadSet(filename='../Datasets/CIFAR-10/adv.txt', image_dir='../Datasets/CIFAR-10/clean_png/', count=50000)
    # trs = rs.get_train_set()
    # tes = rs.get_test_set()
    # print(len(trs), len(tes))
    read_images('../Datasets/CIFAR-10/adv/')
