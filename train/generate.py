import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from attack.fgsm import fgsm_attack
from model.ResNet import resnet20_cifar
from utils.file import ReadSet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 获取数据
# 严格按顺序读取文件,因为要生成跟clean数据集原数据顺序一一对应的图片
read_set = ReadSet(filename='../Datasets/CIFAR-10/clean_label.txt', image_dir='../Datasets/CIFAR-10/clean/', shuffle=False)
cl_train_data = read_set.get_train_set()
sort_loader = DataLoader(cl_train_data, batch_size=100, shuffle=False, num_workers=2)

toPIL = transforms.ToPILImage()

# 本批次开始的文件名
count = 0

if __name__ == "__main__":
    # 加载模型
    net = resnet20_cifar().to(device)

    for images, labels in sort_loader:
        images, labels = images.to(device), labels.to(device)
        images = fgsm_attack(net, device, images, labels, 0.15)
        # 保存图片
        for img in images:
            pic = toPIL(img)
            pic.save('../Datasets/gen_adv/fgsm_0.03/%d.jpg' % count)
            print('生成fgsm对抗样本，epsilon=0.03：%d.jpg' % count)
            count += 1
