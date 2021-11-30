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


# DCT 变换
def image_dct(filename, pixel):
    cv_image = cv2.imread(filename)  # 读取图片，
    b, g, r = cv2.split(cv_image)

    bf = b.astype('float')
    gf = g.astype('float')
    rf = r.astype('float')
    dct_b = cv2.dct(bf)
    dct_g = cv2.dct(gf)
    dct_r = cv2.dct(rf)
    # 图像压缩
    dep_b, dep_g, dep_r = np.zeros(b.shape), np.zeros(g.shape), np.zeros(r.shape)
    dep_b[0:pixel, 0:pixel] = dct_b[0:pixel, 0:pixel]
    dep_g[0:pixel, 0:pixel] = dct_g[0:pixel, 0:pixel]
    dep_r[0:pixel, 0:pixel] = dct_r[0:pixel, 0:pixel]
    # DCT逆变换
    db = cv2.idct(dep_b)
    dg = cv2.idct(dep_g)
    dr = cv2.idct(dep_r)
    # 合并三通道
    img_merge = cv2.merge([dg, dg, dr]).astype(np.uint8)
    source = cv2.merge([b, g, r])
    # OpenCV转PIL
    pil_img = Image.fromarray(cv2.cvtColor(img_merge, cv2.COLOR_BGR2RGB))
    return pil_img


def default_loader(filename, transform=None, dct=False):
    if dct:
        img_pil = image_dct(filename, 26)
    else:
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
