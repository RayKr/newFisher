from torch.utils.data import DataLoader

from utils.file import ReadSet

BATCH_SIZE = 120

# 读取数据
# cifar_train = torchvision.datasets.CIFAR10(root='../Datasets/', train=True, download=True, transform=transform_train)
# cifar_test = torchvision.datasets.CIFAR10(root='../Datasets/', train=False, download=True, transform=transform_train)
# cl_train_loader = DataLoader(cifar_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
# cl_test_loader = DataLoader(cifar_test, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
read_set = ReadSet(filename='../Datasets/CIFAR-10/clean_label.txt', image_dir='../Datasets/CIFAR-10/clean_png/', count=50000)
cl_train_data = read_set.get_train_set()
cl_test_data = read_set.get_test_set()
cl_train_loader = DataLoader(cl_train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
cl_test_loader = DataLoader(cl_test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# pdg生成对抗样本
pgd_set = ReadSet(filename='../Datasets/CIFAR-10/clean_label.txt', image_dir='../Datasets/gen_adv/pgd/', count=50000)
pgd_train_data = pgd_set.get_train_set()
pgd_test_data = pgd_set.get_test_set()
pgd_train_loader = DataLoader(pgd_train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
pgd_test_loader = DataLoader(pgd_test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# fgsm生成的对抗样本
fgsm_set = ReadSet(filename='../Datasets/CIFAR-10/clean_label.txt', image_dir='../Datasets/gen_adv/fgsm_0.1/', count=50000)
fgsm_train_data = fgsm_set.get_train_set()
fgsm_test_data = fgsm_set.get_test_set()
fgsm_train_loader = DataLoader(fgsm_train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
fgsm_test_loader = DataLoader(fgsm_test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# 对抗样本验证集
adv_set = ReadSet(filename='../Datasets/CIFAR-10/adv.txt', image_dir='../Datasets/CIFAR-10/adv/', shuffle=False)
adv_data = adv_set.get_train_set()
adv_loader = DataLoader(adv_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

