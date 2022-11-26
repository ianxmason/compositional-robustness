import numpy as np
import torch
import torch.nn as nn


# DTN network from https://github.com/thuml/CDAN/blob/master/pytorch/network.py
# The code is associated with https://arxiv.org/pdf/1705.10667.pdf
class DTN(nn.Module):
    def __init__(self, n_classes):
        super(DTN, self).__init__()
        # Todo: (maybe/opt) - set num filters, batch norm size etc. dynamically based on number of corruptions/equivariant
        # hooks. Atm is all done manually - e.g. set first layer to 32 to use one hook, then batch norm 64 afterwards etc.
        self.conv_params = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.3),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.5),
            nn.ReLU()
        )

        self.fc_params = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

        self.classifier = nn.Linear(512, n_classes)

    def forward(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        x = self.fc_params(x)
        y = self.classifier(x)
        return y


# Todo: if eqv filters continues - make this more flexible so can set number of filters to train and number
#       of equivariant_hooks to use.
class DTN_half(nn.Module):
    def __init__(self, n_classes):
        super(DTN_half, self).__init__()
        self.conv_params = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.3),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.5),
            nn.ReLU()
        )

        self.fc_params = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

        self.classifier = nn.Linear(512, n_classes)

    def forward(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        x = self.fc_params(x)
        y = self.classifier(x)
        return y


class DTN_Part_One(nn.Module):
    def __init__(self):
        super(DTN_Part_One, self).__init__()
        # Todo: (maybe/opt) - set num filters, batch norm size etc. dynamically based on number of corruptions/equivariant
        # hooks. Atm is all done manually - e.g. set first layer to 32 to use one hook, then batch norm 64 afterwards etc.
        self.conv_params = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.3),
            nn.ReLU()
        )

    def forward(self, x):
        y = self.conv_params(x)
        return y


class DTN_Part_Two(nn.Module):
    def __init__(self, n_classes):
        super(DTN_Part_Two, self).__init__()
        # Todo: (maybe/opt) - set num filters, batch norm size etc. dynamically based on number of corruptions/equivariant
        # hooks. Atm is all done manually - e.g. set first layer to 32 to use one hook, then batch norm 64 afterwards etc.
        self.conv_params = nn.Sequential(
            # nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            # nn.BatchNorm2d(128),
            # nn.Dropout2d(0.3),
            # nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.5),
            nn.ReLU()
        )

        self.fc_params = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

        self.classifier = nn.Linear(512, n_classes)

    def forward(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        x = self.fc_params(x)
        y = self.classifier(x)
        return y


class Filter_Bank(nn.Module):
    def __init__(self):
        super(Filter_Bank, self).__init__()
        # Todo: (maybe/opt) - set num filters, batch norm size etc. dynamically based on number of corruptions/equivariant
        # hooks. Atm is all done manually - e.g. set first layer to 32 to use one hook, then batch norm 64 afterwards etc.
        self.conv_params = nn.Sequential(
            # nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            # nn.BatchNorm2d(64),
            # nn.Dropout2d(0.3),
            # nn.ReLU(),
            # nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            # nn.BatchNorm2d(64),
            # nn.Dropout2d(0.3),
            # nn.ReLU()

            nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.3),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.3),
            nn.ReLU()
        )

    def forward(self, x):
        y = self.conv_params(x)
        return y


class Filter_Bank_Before_Net(nn.Module):
    def __init__(self):
        super(Filter_Bank_Before_Net, self).__init__()
        # Todo: optional - make this a mini autoencoder in terms of shape - eg. 3 channels -> 64 channels -> 3 channels
        # Todo: use dropout, batch norm etc???
        # Todo: combine the two filter bank set ups so we can just take shapes of layers as inpots
        self.conv_params = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.3),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(3),
            nn.Dropout2d(0.1),
            nn.ReLU()
        )

    def forward(self, x):
        y = self.conv_params(x)
        return y


class Filter_Bank_Resad(nn.Module):
    def __init__(self):
        super(Filter_Bank_Resad, self).__init__()
        self.conv_params = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2)
        )

    def forward(self, x):
        y = self.conv_params(x)
        return y


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

    # Extra block for more capacity. If used changed next block from 256 * 4 * 4 to 256 * 2 * 2.
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
            nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2),
            nn.Dropout2d(0.3),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Dropout2d(0.3),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Dropout2d(0.1),
            nn.Tanh()  # normalized data is in range [-1, 1]
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
