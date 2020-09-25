from .cond_resnet import CondCifarResNet


def get_net(arch, **kwargs):
    if arch == "cond_cifar_resnet":
        return CondCifarResNet(**kwargs)
