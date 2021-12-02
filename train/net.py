from model.ResNet import resnet32_cifar
from model.swin_transformer import SwinTransformer

pre_model_path = {
    'pre': './net/pre_train/net_070.pth',
    'clean': './net/clean/net_100.pth',
    'fgsm': './net/fgsm/net_170.pth',
    'rfgsm': './net/pre_rfgsm/net_125_best.pth',
    'pgd': './net/pgd/net_150.pth',
    'mixed': './net/mixed/net_170.pth',
    'swin-t': './net/swin_t/net_100.pth',
}

model_type = {
    'resnet32': resnet32_cifar(),
    'swin-t': SwinTransformer(),
}
