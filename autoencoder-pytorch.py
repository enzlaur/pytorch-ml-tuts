# Import OS
import os
# Import Pytorch
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
# Import Numpy
import numpy as np

def load_all_ecg_file(ecg_file):
    """
    Unfortunately, the generated files already have sampling frequency into it but not normalized (sad)
    """
    ecg_data = np.load(ecg_file)
    ecg_data = np.reshape( ecg_data, newshape=(ecg_data.shape[0], ecg_data.shape[1]) )
    # ecg_data = normalize_by_chunks(ecg_data)
    ecg_data = np.reshape( ecg_data, newshape=(ecg_data.shape[0], ecg_data.shape[1], 1) )
    return ecg_data

class autoencoder( nn.Module ):
    pass