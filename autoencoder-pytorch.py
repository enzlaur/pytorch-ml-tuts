import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, random_split

# Summary-like from Tensorflow
from torchsummary import summary

# Import Numpy
import numpy as np
# Plot Import
import matplotlib.pyplot as plt
# For timer
import time
# Import OS
import os
# import local libs
import ecg_tools_lite as et

from statistics import mean

# ---- SAFE MODEL ----
kernel_size = 16
padding_size= int( (kernel_size/2) ) # If odd, add -1

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 40, kernel_size, stride=2, padding=7 ), nn.ELU(True), # 521
            nn.BatchNorm1d(40),
            nn.Conv1d(40, 20, kernel_size, stride=2, padding=7 ), nn.ELU(True), # 521
            nn.BatchNorm1d(20),
            nn.Conv1d(20, 20, kernel_size, stride=2, padding=7 ), nn.ELU(True), # 521
            nn.BatchNorm1d(20),
            nn.Conv1d(20, 20, kernel_size, stride=2, padding=7 ), nn.ELU(True), # 521
            nn.BatchNorm1d(20),
            nn.Conv1d(20, 40, kernel_size, stride=2, padding=8 ), nn.ELU(True), # 521
            nn.BatchNorm1d(40),
            nn.Conv1d(40, 1, kernel_size, stride=1, padding=7 ), nn.ELU(True), # 521
            nn.BatchNorm1d(1),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(1, 40, kernel_size, stride=1, padding=7 ), nn.ELU(True), # 521
            nn.BatchNorm1d(40),
            nn.ConvTranspose1d(40, 20, kernel_size, stride=2, padding=8 ), nn.ELU(True), # 521
            nn.BatchNorm1d(20),
            nn.ConvTranspose1d(20, 20, kernel_size, stride=2, padding=7 ), nn.ELU(True), # 521
            nn.BatchNorm1d(20),
            nn.ConvTranspose1d(20, 20, kernel_size, stride=2, padding=7 ), nn.ELU(True), # 521
            nn.BatchNorm1d(20),
            nn.ConvTranspose1d(20, 40, kernel_size, stride=2, padding=7 ), nn.ELU(True), # 521
            nn.BatchNorm1d(40),
            nn.ConvTranspose1d(40, 1, kernel_size, stride=2, padding=7 ), nn.ELU(True), # 521
            nn.BatchNorm1d(1),
            # nn.ConvTranspose1d(1, 1, kernel_size, stride=1, padding=7 ), nn.ELU(True), # 521
            # nn.BatchNorm1d(1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x