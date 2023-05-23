import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18, inception_v3
from copy import deepcopy


class SimpleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, batch_norm=True, dropout=0.0,
                 activation=nn.ReLU()):
        super(SimpleConvBlock, self).__init__()

        if batch_norm:
            self.conv_params = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.Dropout2d(dropout),
                activation
            )
        else:
            self.conv_params = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.Dropout2d(dropout),
                activation
            )

    def forward(self, x):
        return self.conv_params(x)


class SimpleFullyConnectedBlock(nn.Module):
    def __init__(self, in_features, out_features, batch_norm=True, dropout=0.0, activation=nn.ReLU()):
        super(SimpleFullyConnectedBlock, self).__init__()

        if batch_norm:
            self.fc_params = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features),
                nn.Dropout(dropout),
                activation
            )
        else:
            self.fc_params = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.Dropout(dropout),
                activation
            )

    def forward(self, x):
        return self.fc_params(x)


class SimpleClassifier(nn.Module):
    def __init__(self, in_features, out_features):
        super(SimpleClassifier, self).__init__()

        self.classifier = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.classifier(x)


# Based on the DTN network from https://github.com/thuml/CDAN/blob/master/pytorch/network.py
# The code is associated with https://arxiv.org/pdf/1705.10667.pdf
def create_emnist_network(total_n_classes, experiment, corruption_names, dev):
    network_blocks = []
    network_block_ckpt_names = []

    network_blocks.append(
        nn.Sequential(
            SimpleConvBlock(3, 64, kernel_size=5, stride=2, padding=2, batch_norm=False, dropout=0.1)  # 0.1
        ).to(dev)
    )
    network_block_ckpt_names.append("{}_ConvBlock1_{}.pt".format(experiment, '-'.join(corruption_names)))

    network_blocks.append(
        nn.Sequential(
            SimpleConvBlock(64, 128, kernel_size=5, stride=2, padding=2, batch_norm=False, dropout=0.3)  # 0.3
        ).to(dev)
    )
    network_block_ckpt_names.append("{}_ConvBlock2_{}.pt".format(experiment, '-'.join(corruption_names)))

    network_blocks.append(
        nn.Sequential(
            SimpleConvBlock(128, 256, kernel_size=5, stride=2, padding=2, batch_norm=False, dropout=0.5)  # 0.5
        ).to(dev)
    )
    network_block_ckpt_names.append("{}_ConvBlock3_{}.pt".format(experiment, '-'.join(corruption_names)))

    network_blocks.append(
        nn.Sequential(
            SimpleConvBlock(256, 256, kernel_size=5, stride=2, padding=2, batch_norm=False, dropout=0.5)  # 0.5
        ).to(dev)
    )
    network_block_ckpt_names.append("{}_ConvBlock4_{}.pt".format(experiment, '-'.join(corruption_names)))

    network_blocks.append(
        nn.Sequential(
            nn.Flatten(),  # Flattens everything except the batch dimension by default
            SimpleFullyConnectedBlock(256 * 2 * 2, 512, batch_norm=False, dropout=0.5)
        ).to(dev)
    )
    network_block_ckpt_names.append("{}_FullyConnected_{}.pt".format(experiment, '-'.join(corruption_names)))

    network_blocks.append(
        nn.Sequential(
            SimpleClassifier(512, total_n_classes)
        ).to(dev)
    )
    network_block_ckpt_names.append("{}_Classifier_{}.pt".format(experiment, '-'.join(corruption_names)))

    assert len(network_block_ckpt_names) == len(network_blocks)
    return network_blocks, network_block_ckpt_names


def create_emnist_modules(experiment, corruption_names, dev):
    modules = []
    module_ckpt_names = []

    modules.append(
        nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            nn.Dropout2d(0.1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2),
            nn.Dropout2d(0.3),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Dropout2d(0.3),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Tanh()  # data is in range [-1, 1]
        ).to(dev)
    )
    module_ckpt_names.append("{}_ConvModule0_{}.pt".format(experiment, '-'.join(corruption_names)))  # In image space

    modules.append(
        nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.Dropout2d(0.3),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.Dropout2d(0.3),
            nn.ReLU()
        ).to(dev)
    )
    module_ckpt_names.append("{}_ConvModule1_{}.pt".format(experiment, '-'.join(corruption_names)))

    modules.append(
        nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2),
            nn.Dropout2d(0.3),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2),
            nn.Dropout2d(0.3),
            nn.ReLU()
        ).to(dev)
    )
    module_ckpt_names.append("{}_ConvModule2_{}.pt".format(experiment, '-'.join(corruption_names)))

    modules.append(
        nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=5, stride=1, padding=2),
            nn.Dropout2d(0.5),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=5, stride=1, padding=2),
            nn.Dropout2d(0.5),
            nn.ReLU()
        ).to(dev)
    )
    module_ckpt_names.append("{}_ConvModule3_{}.pt".format(experiment, '-'.join(corruption_names)))

    modules.append(
        nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=5, stride=1, padding=2),
            nn.Dropout2d(0.5),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=5, stride=1, padding=2),
            nn.Dropout2d(0.5),
            nn.ReLU()
        ).to(dev)
    )
    module_ckpt_names.append("{}_ConvModule4_{}.pt".format(experiment, '-'.join(corruption_names)))

    modules.append(
        nn.Sequential(
            nn.Linear(512, 512),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.Dropout(0.5),
            nn.ReLU()
        ).to(dev)
    )
    module_ckpt_names.append("{}_FullyConnectedModule5_{}.pt".format(experiment, '-'.join(corruption_names)))

    assert len(module_ckpt_names) == len(modules)
    return modules, module_ckpt_names


def create_emnist_autoencoder(experiment, corruption_names, dev):
    network_blocks = []
    network_block_ckpt_names = []

    network_blocks.append(
        nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            nn.Dropout2d(0.1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.Dropout2d(0.3),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.Dropout2d(0.5),
            nn.ReLU()
        ).to(dev)
    )
    network_block_ckpt_names.append("{}_Encoder_{}.pt".format(experiment, '-'.join(corruption_names)))

    network_blocks.append(
        nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2),
            nn.Dropout2d(0.5),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Dropout2d(0.3),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Tanh()  # data is in range [-1, 1]
        ).to(dev)
    )
    network_block_ckpt_names.append("{}_Decoder_{}.pt".format(experiment, '-'.join(corruption_names)))

    assert len(network_block_ckpt_names) == len(network_blocks)
    return network_blocks, network_block_ckpt_names


def create_cifar_network(total_n_classes, experiment, corruption_names, dev):
    resnet = resnet18(pretrained=False)

    network_blocks = []
    network_block_ckpt_names = []
    # Input 3, 32, 32
    network_blocks.append(nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu).to(dev))  # Remove max pooling
    network_block_ckpt_names.append("{}_ResnetBlock1_{}.pt".format(experiment, '-'.join(corruption_names)))
    # 64, 16, 16
    network_blocks.append(resnet.layer1[0].to(dev))
    network_block_ckpt_names.append("{}_ResnetBlock2_{}.pt".format(experiment, '-'.join(corruption_names)))
    # 64, 16, 16
    network_blocks.append(resnet.layer1[1].to(dev))
    network_block_ckpt_names.append("{}_ResnetBlock3_{}.pt".format(experiment, '-'.join(corruption_names)))
    # 64, 16, 16
    network_blocks.append(resnet.layer2[0].to(dev))
    network_block_ckpt_names.append("{}_ResnetBlock4_{}.pt".format(experiment, '-'.join(corruption_names)))
    # 128, 8, 8
    network_blocks.append(resnet.layer2[1].to(dev))
    network_block_ckpt_names.append("{}_ResnetBlock5_{}.pt".format(experiment, '-'.join(corruption_names)))
    # 128, 8, 8
    network_blocks.append(resnet.layer3[0].to(dev))
    network_block_ckpt_names.append("{}_ResnetBlock6_{}.pt".format(experiment, '-'.join(corruption_names)))
    # 256, 4, 4
    network_blocks.append(resnet.layer3[1].to(dev))
    network_block_ckpt_names.append("{}_ResnetBlock7_{}.pt".format(experiment, '-'.join(corruption_names)))
    # 256, 4, 4
    network_blocks.append(resnet.layer4[0].to(dev))
    network_block_ckpt_names.append("{}_ResnetBlock8_{}.pt".format(experiment, '-'.join(corruption_names)))
    # 512, 2, 2
    network_blocks.append(nn.Sequential(resnet.layer4[1], resnet.avgpool, nn.Flatten()).to(dev))
    network_block_ckpt_names.append("{}_ResnetBlock9_{}.pt".format(experiment, '-'.join(corruption_names)))
    # 512,
    network_blocks.append(SimpleClassifier(512, total_n_classes).to(dev))
    network_block_ckpt_names.append("{}_ResnetClassifier_{}.pt".format(experiment, '-'.join(corruption_names)))
    # 10,
    assert len(network_block_ckpt_names) == len(network_blocks)
    return network_blocks, network_block_ckpt_names


def create_cifar_modules(experiment, corruption_names, dev):
    resnet = resnet18(pretrained=False)

    modules = []
    module_ckpt_names = []

    modules.append(
        nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=2)
            # data has mean 0, std 1. so no activation
        ).to(dev)
    )
    module_ckpt_names.append("{}_ResnetModule0_{}.pt".format(experiment, '-'.join(corruption_names)))  # In image space

    modules.append(deepcopy(resnet.layer1[1]).to(dev))
    module_ckpt_names.append("{}_ResnetModule1_{}.pt".format(experiment, '-'.join(corruption_names)))

    modules.append(deepcopy(resnet.layer1[1]).to(dev))
    module_ckpt_names.append("{}_ResnetModule2_{}.pt".format(experiment, '-'.join(corruption_names)))

    modules.append(deepcopy(resnet.layer1[1]).to(dev))
    module_ckpt_names.append("{}_ResnetModule3_{}.pt".format(experiment, '-'.join(corruption_names)))

    modules.append(deepcopy(resnet.layer2[1]).to(dev))
    module_ckpt_names.append("{}_ResnetModule4_{}.pt".format(experiment, '-'.join(corruption_names)))

    modules.append(deepcopy(resnet.layer2[1]).to(dev))
    module_ckpt_names.append("{}_ResnetModule5_{}.pt".format(experiment, '-'.join(corruption_names)))

    modules.append(deepcopy(resnet.layer3[1]).to(dev))
    module_ckpt_names.append("{}_ResnetModule6_{}.pt".format(experiment, '-'.join(corruption_names)))

    modules.append(deepcopy(resnet.layer3[1]).to(dev))
    module_ckpt_names.append("{}_ResnetModule7_{}.pt".format(experiment, '-'.join(corruption_names)))

    modules.append(deepcopy(resnet.layer4[1]).to(dev))
    module_ckpt_names.append("{}_ResnetModule8_{}.pt".format(experiment, '-'.join(corruption_names)))

    modules.append(
        nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        ).to(dev)
    )
    module_ckpt_names.append("{}_ResnetModule9_{}.pt".format(experiment, '-'.join(corruption_names)))

    assert len(module_ckpt_names) == len(modules)
    return modules, module_ckpt_names


def create_cifar_autoencoder(experiment, corruption_names, dev):
    network_blocks = []
    network_block_ckpt_names = []

    network_blocks.append(
        nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU()
        ).to(dev)
    )
    network_block_ckpt_names.append("{}_Encoder_{}.pt".format(experiment, '-'.join(corruption_names)))

    network_blocks.append(
        nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=2)
            # data has mean 0, std 1. so no activation
        ).to(dev)
    )
    network_block_ckpt_names.append("{}_Decoder_{}.pt".format(experiment, '-'.join(corruption_names)))

    assert len(network_block_ckpt_names) == len(network_blocks)
    return network_blocks, network_block_ckpt_names


def create_facescrub_network(total_n_classes, experiment, corruption_names, dev):
    # https://stackoverflow.com/questions/57421842/image-size-of-256x256-not-299x299-fed-into-inception-v3-model-pytorch-and-wo
    # we can use facescrub images of size 100x100 if we don't use aux_logits
    # (no auxiliary classifier as in section 4 https://arxiv.org/pdf/1512.00567.pdf)
    inception = inception_v3(pretrained=False)

    network_blocks = []
    network_block_ckpt_names = []
    # Input 3, 100, 100
    network_blocks.append(inception.Conv2d_1a_3x3.to(dev))
    network_block_ckpt_names.append("{}_InceptionBlock1_{}.pt".format(experiment, '-'.join(corruption_names)))
    # 32, 49, 49
    network_blocks.append(inception.Conv2d_2a_3x3.to(dev))
    network_block_ckpt_names.append("{}_InceptionBlock2_{}.pt".format(experiment, '-'.join(corruption_names)))
    # 32, 47, 47
    network_blocks.append(inception.Conv2d_2b_3x3.to(dev))
    network_block_ckpt_names.append("{}_InceptionBlock3_{}.pt".format(experiment, '-'.join(corruption_names)))
    # 64, 47, 47
    network_blocks.append(nn.Sequential(inception.maxpool1, inception.Conv2d_3b_1x1).to(dev))
    network_block_ckpt_names.append("{}_InceptionBlock4_{}.pt".format(experiment, '-'.join(corruption_names)))
    # 80, 23, 23
    network_blocks.append(inception.Conv2d_4a_3x3.to(dev))
    network_block_ckpt_names.append("{}_InceptionBlock5_{}.pt".format(experiment, '-'.join(corruption_names)))
    # 192, 21, 21
    network_blocks.append(nn.Sequential(inception.maxpool2, inception.Mixed_5b).to(dev))
    network_block_ckpt_names.append("{}_InceptionBlock6_{}.pt".format(experiment, '-'.join(corruption_names)))
    # 256, 10, 10
    network_blocks.append(inception.Mixed_5c.to(dev))
    network_block_ckpt_names.append("{}_InceptionBlock7_{}.pt".format(experiment, '-'.join(corruption_names)))
    # 288, 10, 10
    network_blocks.append(inception.Mixed_5d.to(dev))
    network_block_ckpt_names.append("{}_InceptionBlock8_{}.pt".format(experiment, '-'.join(corruption_names)))
    # 288, 10, 10
    network_blocks.append(inception.Mixed_6a.to(dev))
    network_block_ckpt_names.append("{}_InceptionBlock9_{}.pt".format(experiment, '-'.join(corruption_names)))
    # 768, 4, 4
    network_blocks.append(inception.Mixed_6b.to(dev))
    network_block_ckpt_names.append("{}_InceptionBlock10_{}.pt".format(experiment, '-'.join(corruption_names)))
    # 768, 4, 4
    network_blocks.append(inception.Mixed_6c.to(dev))
    network_block_ckpt_names.append("{}_InceptionBlock11_{}.pt".format(experiment, '-'.join(corruption_names)))
    # 768, 4, 4
    network_blocks.append(inception.Mixed_6d.to(dev))
    network_block_ckpt_names.append("{}_InceptionBlock12_{}.pt".format(experiment, '-'.join(corruption_names)))
    # 768, 4, 4
    network_blocks.append(inception.Mixed_6e.to(dev))
    network_block_ckpt_names.append("{}_InceptionBlock13_{}.pt".format(experiment, '-'.join(corruption_names)))
    # 768, 4, 4
    network_blocks.append(inception.Mixed_7a.to(dev))
    network_block_ckpt_names.append("{}_InceptionBlock14_{}.pt".format(experiment, '-'.join(corruption_names)))
    # 1280, 1, 1
    network_blocks.append(inception.Mixed_7b.to(dev))
    network_block_ckpt_names.append("{}_InceptionBlock15_{}.pt".format(experiment, '-'.join(corruption_names)))
    # 2048, 1, 1
    network_blocks.append(nn.Sequential(inception.Mixed_7c, inception.avgpool, nn.Flatten()).to(dev))
    network_block_ckpt_names.append("{}_InceptionBlock16_{}.pt".format(experiment, '-'.join(corruption_names)))
    # 2048,
    network_blocks.append(SimpleClassifier(2048, total_n_classes).to(dev))
    network_block_ckpt_names.append("{}_InceptionClassifier_{}.pt".format(experiment, '-'.join(corruption_names)))
    # 388,
    assert len(network_block_ckpt_names) == len(network_blocks)
    return network_blocks, network_block_ckpt_names


def create_facescrub_modules(experiment, corruption_names, dev):
    # We use existing inception blocks as modules where possible. Elsewhere, use simple convolutional layers.
    inception = inception_v3(pretrained=False)

    modules = []
    module_ckpt_names = []

    modules.append(
        nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=5, stride=1, padding=2)
            # data has mean 0, std 1. so no activation
        ).to(dev)
    )
    module_ckpt_names.append("{}_InceptionModule0_{}.pt".format(experiment, '-'.join(corruption_names)))  # In image space

    modules.append(
        nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        ).to(dev)
    )
    module_ckpt_names.append("{}_InceptionModule1_{}.pt".format(experiment, '-'.join(corruption_names)))

    modules.append(
        nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        ).to(dev)
    )
    module_ckpt_names.append("{}_InceptionModule2_{}.pt".format(experiment, '-'.join(corruption_names)))

    modules.append(
        nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU()
        ).to(dev)
    )
    module_ckpt_names.append("{}_InceptionModule3_{}.pt".format(experiment, '-'.join(corruption_names)))

    modules.append(
        nn.Sequential(
            nn.Conv2d(80, 80, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(80),
            nn.ReLU(),
            nn.Conv2d(80, 80, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(80),
            nn.ReLU(),
            nn.Conv2d(80, 80, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(80),
            nn.ReLU()
        ).to(dev)
    )
    module_ckpt_names.append("{}_InceptionModule4_{}.pt".format(experiment, '-'.join(corruption_names)))

    modules.append(
        nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Conv2d(192, 192, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Conv2d(192, 192, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU()
        ).to(dev)
    )
    module_ckpt_names.append("{}_InceptionModule5_{}.pt".format(experiment, '-'.join(corruption_names)))

    modules.append(
        nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU()
        ).to(dev)
    )
    module_ckpt_names.append("{}_InceptionModule6_{}.pt".format(experiment, '-'.join(corruption_names)))

    modules.append(deepcopy(inception.Mixed_5d).to(dev))  # 288
    module_ckpt_names.append("{}_InceptionModule7_{}.pt".format(experiment, '-'.join(corruption_names)))

    modules.append(deepcopy(inception.Mixed_5d).to(dev))  # 288
    module_ckpt_names.append("{}_InceptionModule8_{}.pt".format(experiment, '-'.join(corruption_names)))

    modules.append(deepcopy(inception.Mixed_6b).to(dev))  # 768
    module_ckpt_names.append("{}_InceptionModule9_{}.pt".format(experiment, '-'.join(corruption_names)))

    modules.append(deepcopy(inception.Mixed_6c).to(dev))  # 768
    module_ckpt_names.append("{}_InceptionModule10_{}.pt".format(experiment, '-'.join(corruption_names)))

    modules.append(deepcopy(inception.Mixed_6d).to(dev))  # 768
    module_ckpt_names.append("{}_InceptionModule11_{}.pt".format(experiment, '-'.join(corruption_names)))

    modules.append(deepcopy(inception.Mixed_6e).to(dev))  # 768
    module_ckpt_names.append("{}_InceptionModule12_{}.pt".format(experiment, '-'.join(corruption_names)))

    modules.append(deepcopy(inception.Mixed_6e).to(dev))  # 768
    module_ckpt_names.append("{}_InceptionModule13_{}.pt".format(experiment, '-'.join(corruption_names)))

    modules.append(
        nn.Sequential(
            nn.Conv2d(1280, 1280, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(1280),
            nn.ReLU(),
            nn.Conv2d(1280, 1280, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(1280),
            nn.ReLU(),
            nn.Conv2d(1280, 1280, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(1280),
            nn.ReLU()
        ).to(dev)
    )
    module_ckpt_names.append("{}_InceptionModule14_{}.pt".format(experiment, '-'.join(corruption_names)))

    modules.append(deepcopy(inception.Mixed_7c).to(dev))  # 2048
    module_ckpt_names.append("{}_InceptionModule15_{}.pt".format(experiment, '-'.join(corruption_names)))

    modules.append(
        nn.Sequential(
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU()
        ).to(dev)
    )
    module_ckpt_names.append("{}_InceptionModule16_{}.pt".format(experiment, '-'.join(corruption_names)))

    assert len(module_ckpt_names) == len(modules)
    return modules, module_ckpt_names


def create_facescrub_autoencoder(experiment, corruption_names, dev):
    network_blocks = []
    network_block_ckpt_names = []

    network_blocks.append(
        nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU()
        ).to(dev)
    )
    network_block_ckpt_names.append("{}_Encoder_{}.pt".format(experiment, '-'.join(corruption_names)))

    network_blocks.append(
        nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=5, stride=2, padding=2, output_padding=1)
            # data has mean 0, std 1. so no activation
        ).to(dev)
    )
    network_block_ckpt_names.append("{}_Decoder_{}.pt".format(experiment, '-'.join(corruption_names)))

    assert len(network_block_ckpt_names) == len(network_blocks)
    return network_blocks, network_block_ckpt_names


