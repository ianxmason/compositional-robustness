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
