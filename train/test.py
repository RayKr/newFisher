import torch

from data import cl_test_loader, adv_loader
from eval import eval_acc
from net import model_type, pre_model_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def attack_test(model_name, pre_model_name, desc=None, attack_method=None, **kwargs):
    net = model_type[model_name].to(device)
    path = pre_model_path[pre_model_name]
    net.load_state_dict(torch.load(path))
    net.eval()

    if desc is not None:
        print(desc)
    clean_acc = eval_acc(cl_test_loader, net, device, attack_method=attack_method, **kwargs)
    adv_acc = eval_acc(adv_loader, net, device, attack_method=attack_method, **kwargs)
    print('Clean_Acc = %.3f%%, Adv_Acc = %.3f%%' % (clean_acc, adv_acc))
    # fgsm_acc = eval_attack_acc(fgsm_test_loader, net, device, attack_method=attack_method, **kwargs)
    # print('Generated FGSM Adversarial Examples Acc = %.3f%%' % fgsm_acc)
    # pgd_acc = eval_attack_acc(pgd_test_loader, net, device, attack_method=attack_method, **kwargs)
    # print('Generated PGD Adversarial Examples Acc = %.3f%%' % pgd_acc)


def robust_test(model_name='resnet32', pre_model_name='clean', model_title='Clean预训练模型'):
    print(f'--------------{model_title}--------------')
    attack_test(model_name, pre_model_name, '【无攻击】')
    attack_test(model_name, pre_model_name, '【FGSM攻击：epsilon=0.1】', attack_method='fgsm', epsilon=0.1)
    attack_test(model_name, pre_model_name, '【PGD攻击：epsilon=0.3, alpha=2/255, iters=20】', attack_method='pgd', epsilon=0.3, alpha=2/255, iters=20)
    attack_test(model_name, pre_model_name, '【RFGSM攻击：alpha=0.1, eps=0.5】', attack_method='rfgsm', alpha=0.1, eps=0.5)
    attack_test(model_name, pre_model_name, '【FGM攻击：epsilon=0.15】', attack_method='fgm', epsilon=0.15)


if __name__ == "__main__":
    # robust_test(model_name='resnet32', pre_model_name='pre', model_title='预训练模型')
    robust_test(model_name='resnet32', pre_model_name='clean', model_title='Clean训练模型')
    # robust_test(model_name='resnet32', pre_model_name='fgsm', model_title='FGSM对抗训练模型')
    # robust_test(model_name='resnet32', pre_model_name='rfgsm', model_title='RFGSM对抗训练模型')
    # robust_test(model_name='resnet32', pre_model_name='pgd', model_title='PGD对抗训练模型')
    # robust_test(model_name='swin-t', pre_model_name='swin-t', model_title='Swin Transformer训练模型')
