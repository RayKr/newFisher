# -*-coding: utf-8 -*-
import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import image_processing
import os

from file import read_clean_list


class TorchDataset(Dataset):
    def __init__(self, filename, image_dir, resize_height=None, resize_width=None, repeat=1):
        """
        :param filename: 数据文件TXT：格式：imge_name.jpg label1_id labe2_id
        :param image_dir: 图片路径：image_dir+imge_name.jpg构成图片的完整路径
        :param resize_height 为None时，不进行缩放
        :param resize_width  为None时，不进行缩放，
                              PS：当参数resize_height或resize_width其中一个为None时，可实现等比例缩放
        :param repeat: 所有样本数据重复次数，默认循环一次，当repeat为None时，表示无限循环
        """
        self.image_label_list = read_clean_list(filename)
        self.image_dir = image_dir
        self.len = len(self.image_label_list)
        self.repeat = repeat
        self.resize_height = resize_height
        self.resize_width = resize_width

        # 相关预处理的初始化
        '''class torchvision.transforms.ToTensor'''
        # 把shape=(H,W,C)的像素值范围为[0, 255]的PIL.Image或者numpy.ndarray数据
        # 转换成shape=(C,H,W)的像素数据，并且被归一化到[0.0, 1.0]的torch.FloatTensor类型。
        self.toTensor = transforms.ToTensor()

        '''class torchvision.transforms.Normalize(mean, std)
        此转换类作用于torch. * Tensor,给定均值(R, G, B) 和标准差(R, G, B)，
        用公式channel = (channel - mean) / std进行规范化。
        '''
        # self.normalize=transforms.Normalize()

    def __getitem__(self, i):
        index = i % self.len
        # print("i={},index={}".format(i, index))
        image_name, label = self.image_label_list[index]
        image_path = os.path.join(self.image_dir, image_name)
        img = self.load_data(image_path, self.resize_height, self.resize_width, normalization=False)
        img = self.data_preproccess(img)
        label = np.array(label)
        return img, label

    def __len__(self):
        if self.repeat is None:
            data_len = 10000000
        else:
            data_len = len(self.image_label_list) * self.repeat
        return data_len

    def read_file(self, filename):
        image_label_list = []
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                # rstrip：用来去除结尾字符、空白符(包括\n、\r、\t、' '，即：换行、回车、制表符、空格)
                content = line.rstrip().split(' ')
                name = content[0]
                labels = []
                for value in content[1:]:
                    labels.append(int(value))
                image_label_list.append((name, labels))
        print(image_label_list)
        return image_label_list

    def load_data(self, path, resize_height, resize_width, normalization):
        """
        加载数据
        :param path:
        :param resize_height:
        :param resize_width:
        :param normalization: 是否归一化
        :return:
        """
        image = image_processing.read_image(path, resize_height, resize_width, normalization)
        return image

    def data_preproccess(self, data):
        """
        数据预处理
        :param data:
        :return:
        """
        data = self.toTensor(data)
        return data


if __name__ == '__main__':
    train_filename = "../Datasets/CIFAR-10/adv.txt"
    # test_filename="../dataset/test.txt"
    image_dir = '../Datasets/CIFAR-10/clean'

    epoch_num = 2  # 总样本循环次数
    batch_size = 4  # 训练时的一组数据的大小
    train_data_nums = 10
    max_iterate = int((train_data_nums + batch_size - 1) / batch_size * epoch_num)  # 总迭代次数

    train_data = TorchDataset(filename=train_filename, image_dir=image_dir, repeat=1)
    print(train_data)
    # test_data = TorchDataset(filename=test_filename, image_dir=image_dir,repeat=1)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)
    # test_loader = DataLoader(dataset=test_data, batch_size=batch_size,shuffle=False)

    # [1]使用epoch方法迭代，TorchDataset的参数repeat=1
    for epoch in range(epoch_num):
        for batch_image, batch_label in train_loader:
            print(batch_image, batch_label)
            image = batch_image[0, :]
            image = image.numpy()  # image=np.array(image)
            image = image.transpose(1, 2, 0)  # 通道由[c,h,w]->[h,w,c]
            # image_processing.cv_show_image("image", image)
            # print("batch_image.shape:{},batch_label:{}".format(batch_image.shape, batch_label))
            # batch_x, batch_y = Variable(batch_x), Variable(batch_y)

    '''
    下面两种方式，TorchDataset设置repeat=None可以实现无限循环，退出循环由max_iterate设定
    '''
    # train_data = TorchDataset(filename=train_filename, image_dir=image_dir, repeat=None)
    # train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)
    # [2]第2种迭代方法
    # for step, (batch_image, batch_label) in enumerate(train_loader):
    #     image = batch_image[0, :]
    #     image = image.numpy()  # image=np.array(image)
    #     image = image.transpose(1, 2, 0)  # 通道由[c,h,w]->[h,w,c]
    #     # image_processing.cv_show_image("image", image)
    #     print("step:{},batch_image.shape:{},batch_label:{}".format(step, batch_image.shape, batch_label))
    #     # batch_x, batch_y = Variable(batch_x), Variable(batch_y)
    #     if step >= max_iterate:
    #         break
    # [3]第3种迭代方法
    # for step in range(max_iterate):
    #     batch_image, batch_label=train_loader.__iter__().__next__()
    #     image=batch_image[0,:]
    #     image=image.numpy()#image=np.array(image)
    #     image = image.transpose(1, 2, 0)  # 通道由[c,h,w]->[h,w,c]
    #     image_processing.cv_show_image("image",image)
    #     print("batch_image.shape:{},batch_label:{}".format(batch_image.shape,batch_label))
    #     # batch_x, batch_y = Variable(batch_x), Variable(batch_y)