# Import Tensorflow
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses
# Import plot
import matplotlib.pyplot as plt
# Numpy
import numpy as np
# math
import math
# For test data splitting
from sklearn.model_selection import train_test_split




def get_snr_in(sig_orig, sig_noisy):
    snr = 0.0
    upper = 0.0
    lower = 0.0
    for i in range(sig_orig.shape[0]):
        for j in range(sig_orig.shape[1]):
            upper = upper + (sig_orig[i][j]**2)
    for i in range(sig_noisy.shape[0]):
        for j in range(sig_noisy.shape[1]):
            lower = lower + ((sig_noisy[i][j] - sig_orig[i][j])**2)
    snr = 10 * math.log((upper / lower), 10)
    return snr



# ==== FOR USE WITH UPDATED PYTORCH IMP ====
def get_snr_imp_flat(ecg_noisy, ecg_test):
    snr = 0.0
    upper = 0.0
    lower = 0.0
    
    for i in range( len(ecg_test) ):
        upper = upper + (ecg_test[i]**2)

    for i in range( len(ecg_noisy) ):
        lower = lower + ((ecg_noisy[i] - ecg_test[i])**2)

    snr = 10 * math.log((upper/lower), 10)

    return snr



# ==== DEPRECATED, IGNORE =====
def get_snr_imp_slow(sig_orig, sig_noisy, sig_decoded):
    """
    Very slow, do not use.
    Use get_snr_imp instead
    """
    snr_in = 0.0
    snr_out = 0.0
    snr_imp = 0.0
    upper = 0.0
    lower_in = 0.0
    lower_out = 0.0

    # compute for the summation in the numerator (used for both SNRin and SNRout)
    for i in range(sig_orig.shape[0]):
        for j in range(sig_orig.shape[1]):
            upper = upper + (sig_orig[i][j]**2)
    # Compute for the summation in the denominator (used for SNRin only)
    for i in range(sig_noisy.shape[0]):
        for j in range(sig_noisy.shape[1]):
            lower_in = lower_in + ((sig_noisy[i][j] - sig_orig[i][j])**2)
    # Compute for the summation in the denominator (used for SNRout only)
    for i in range(sig_decoded.shape[0]):
        for j in range(sig_decoded.shape[1]):
            lower_out = lower_out + ((sig_decoded[i][j] - sig_orig[i][j])**2)
    snr_in = 10 * math.log((upper / lower_in), 10)
    snr_out = 10 * math.log((upper / lower_out), 10)
    snr_imp = snr_out - snr_in
    print( str(snr_imp) + '=' + str(snr_out) + '-' + str(snr_in))
    return snr_imp




def get_snr_imp(sig_clean, sig_noisy, sig_decoded, v=False):
    
    snr_in = 0.0
    snr_out = 0.0
    snr_imp = 0.0
    upper = 0.0
    lower_in = 0.0
    lower_out = 0.0
    
    # compute for the summation in the numerator (used for both SNRin and SNRout)
    for i in range(len(sig_clean)):
            upper = upper + (sig_clean[i]**2)
    # Compute for the summation in the denominator (used for SNRin only)
    for i in range(len(sig_noisy)):
            lower_in = lower_in + ((sig_noisy[i] - sig_clean[i])**2)
    # Compute for the summation in the denominator (used for SNRout only)
    for i in range(len(sig_decoded)):
            lower_out = lower_out + ((sig_decoded[i] - sig_clean[i])**2)
    
    snr_in = 10 * math.log((upper / lower_in), 10)
    snr_out = 10 * math.log((upper / lower_out), 10)
    snr_imp = snr_out - snr_in

    snr_in = np.around(snr_in, 2)
    snr_out = np.around(snr_out, 2)
    snr_imp = np.around(snr_imp, 2)

    if v == True:
        print( str(snr_imp) + ' = ' + str(snr_out) + ' - ' + str(snr_in) )
    return snr_imp

def compute_snr_imp(ecg_clean, ecg_noisy, ecg_denoised):
    # iSNR = 10 * log10 ( sum( abs(orig(:) - noisy(:)) ^ 2) / sum( abs(orig(:) - restored(:) ^ 2)))
    numer = 0
    denom = 0
    result = 0

    

    return result
    
def compare_qrs_detect(sig_denoised, sig_noisy):
    sig_denoised = sig_denoised.numpy()
    pass