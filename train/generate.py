import os
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from attack.fgsm import fgsm_attack
from attack.pgd import pgd_attack
from model.ResNet import resnet20_cifar, resnet32_cifar
from utils.file import ReadSet, read_images, TrainSet
from attack.fgsm import fgsm_attack
from attack.pgd import pgd_attack
from attack.rfgsm import rfgsm_attack

attack_list = {'fgsm': fgsm_attack, 'pgd': pgd_attack, 'rfgsm': rfgsm_attack}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 获取数据
# 严格按顺序读取文件,因为要生成跟clean数据集原数据顺序一一对应的图片
read_set = ReadSet(filename='../Datasets/CIFAR-10/clean_label.txt', image_dir='../Datasets/CIFAR-10/clean_png/', shuffle=False)
cl_train_data = read_set.get_train_set()
sort_loader = DataLoader(cl_train_data, batch_size=100, shuffle=False, num_workers=2)

toPIL = transforms.ToPILImage()
# 加载模型
net = resnet32_cifar().to(device)


def generate_jpg(input_dir, output_dir):
    count = 0
    files = os.listdir(input_dir)
    files.sort(key=lambda x: int(x[:-4]))  # 倒着数第四位'.'为分界线，按照‘.’左边的数字从小到大排序
    for _, item in enumerate(files):
        filename = item.split('.')[0]
        img = Image.open(input_dir + item)
        img.save(output_dir + filename + '.jpg', quality=75)
        print(count)
        count += 1


def generate_adv_example(attack_method='fgsm', **kwargs):
    """
    生成对抗样本
    :param attack_method:
    :param kwargs:
    :return:
    """
    count = 0
    for images, labels in sort_loader:
        images, labels = images.to(device), labels.to(device)
        if attack_method:
            att = attack_list[attack_method]
            images = att(net, device, images, labels, **kwargs)
        # 保存图片
        for img in images:
            pic = toPIL(img)
            pic.save('../Datasets/gen_adv/fgsm_0.1/%d.png' % count)
            print('生成pdg对抗样本：%d.png' % count)
            count += 1

if __name__ == "__main__":
    # generate_adv_example()
    generate_jpg(input_dir='../Datasets/predict/up/', output_dir='../Datasets/predict/upJPG/')

