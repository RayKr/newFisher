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


def read_list(filename, count=None, shuffle=True):
    """
    从txt文件里获取训练集和测试集
    根据clean_label.txt的数据分布规律，可以每隔len行随机80%行作为训练集，剩下的作为测试集
    :param shuffle: 乱序
    :param filename: 文件名
    :param count: 划分训练集和测试集的数据，默认80%训练集，20%测试集
    """
    result = []
    with open(filename, "r") as f:
        for line in f.readlines():
            content = line.strip('\n').split(' ')  # 去掉列表中每一个元素的换行符，然后分割
            name = content[0]
            labels = int(content[1])
            result.append((name, labels))
    # 先打乱某一分类的样本的顺序
    if shuffle:
        random.shuffle(result)
    # 再划分
    if count:
        train_list = result[0: count]
        test_list = result[count: int(len(result) + 1)]
        return train_list, test_list
    else:
        return result


def read_images(path):
    list = []
    files = os.listdir(path)
    files.sort(key=lambda x: int(x[:-4]))  # 倒着数第四位'.'为分界线，按照‘.’左边的数字从小到大排序
    for index, item in enumerate(files):
        list.append((item, index))
    return list


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

