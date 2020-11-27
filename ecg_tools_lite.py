# imports
import wfdb
import peakutils
import numpy as np
import pandas as pd
import tensorflow as tf
# froms
from sklearn import preprocessing # for normalizing data
# For test data splitting
from sklearn.model_selection import train_test_split


def load_ecg_file(ecg_file):
    """
    Unfortunately, the generated files already have sampling frequency into it but not normalized (sad)
    """
    ecg_data = np.load(ecg_file)
    ecg_data = np.reshape( ecg_data, newshape=(ecg_data.shape[0], ecg_data.shape[1]) )
    ecg_data = np.reshape( ecg_data, newshape=(ecg_data.shape[0], ecg_data.shape[1], 1) )
    
    return ecg_data

def load_signal(data_name, folder='ecg_data/', v_fields=False, channel=1):
    """
    Loads signals and fields from a ``WFDB.RDSAMP``

    Returns:
        1D array of signals (entire record)
        signals, fields (from wfdb.rdsamp).

        Usage:
            signals, fields = wfdb.rdsamp(...)

    Parameters:

        data_name (str): File name from MIT-BIH (e.g. ``100``, ``124e06``).
        folder (str): folder directory from within this notebook, add '/' at the end (e.g. ```ecg_data```)
        v_fields (bool): True to have function print signal's fields attribute

    Example:
        load_signal('100', 'ecg_data'/)
    """
    file_address = folder + data_name
    signals, fields = wfdb.rdsamp(file_address, channels=[channel])
    if v_fields == True:
        print("Printing fields information of the signal: ")
        print(fields)
    # return signals from the sample
    return signals