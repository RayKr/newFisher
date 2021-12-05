import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.file import ReadSet, read_list, TrainSet

BATCH_SIZE = 128

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
# 准备数据集并预处理
transform_train = transforms.Compose([
    # 训练集上做数据增强
    transforms.RandomCrop(32, padding=4),  # 先四周填充0，在把图像随机裁剪成32*32
    # transforms.Resize([224, 224]),
    transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    normalize,
])

# 读取数据
# 原始数据包
cifar_train = torchvision.datasets.CIFAR10(root='../Datasets/', train=True, download=True, transform=transform_train)
cifar_test = torchvision.datasets.CIFAR10(root='../Datasets/', train=False, download=True, transform=transform_train)
org_train_loader = DataLoader(cifar_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
org_test_loader = DataLoader(cifar_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

# clean样本集
read_set = ReadSet(filename='../Datasets/CIFAR-10/clean_label.txt', image_dir='../Datasets/CIFAR-10/clean_png/', count=50000, transform=transform_train)
cl_train_data = read_set.get_train_set()
cl_test_data = read_set.get_test_set()
cl_train_loader = DataLoader(cl_train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
cl_test_loader = DataLoader(cl_test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

# pdg生成对抗样本
pgd_set = ReadSet(filename='../Datasets/CIFAR-10/clean_label.txt', image_dir='../Datasets/gen_adv/pgd/', count=50000, transform=transform_train)
pgd_train_data = pgd_set.get_train_set()
pgd_test_data = pgd_set.get_test_set()
pgd_train_loader = DataLoader(pgd_train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
pgd_test_loader = DataLoader(pgd_test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# fgsm生成的对抗样本
fgsm_set = ReadSet(filename='../Datasets/CIFAR-10/clean_label.txt', image_dir='../Datasets/gen_adv/fgsm_0.1/', count=50000, transform=transform_train)
fgsm_train_data = fgsm_set.get_train_set()
fgsm_test_data = fgsm_set.get_test_set()
fgsm_train_loader = DataLoader(fgsm_train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
fgsm_test_loader = DataLoader(fgsm_test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# 对抗样本验证集
adv_set = ReadSet(filename='../Datasets/CIFAR-10/adv.txt', image_dir='../Datasets/CIFAR-10/adv/', shuffle=False, transform=transform_train)
adv_data = adv_set.get_train_set()
adv_loader = DataLoader(adv_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# 混合样本
# clean_list, _ = read_list(filename='../Datasets/CIFAR-10/clean_label.txt', image_dir='../Datasets/CIFAR-10/clean_png/', shuffle=True, count=50000)
clean_list, _ = read_list(filename='../Datasets/CIFAR-10/clean_label_jpg.txt', image_dir='../Datasets/CIFAR-10/clean_jpg/', shuffle=True, count=50000)
# pgd_list, _ = read_list(filename='../Datasets/CIFAR-10/clean_label.txt', image_dir='../Datasets/gen_adv/pgd/', shuffle=True, count=10000)
adv_list, _ = read_list(filename='../Datasets/CIFAR-10/adv.txt', image_dir='../Datasets/CIFAR-10/adv/')
mixed_list = clean_list + adv_list + adv_list + adv_list
mixed_data = TrainSet(mixed_list, transform=transform_train)
mixed_loader = DataLoader(mixed_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
