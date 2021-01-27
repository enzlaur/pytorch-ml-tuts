# imports
import wfdb
import peakutils
import numpy as np
import pandas as pd
# froms
from sklearn import preprocessing # for normalizing data
# For test data splitting
from sklearn.model_selection import train_test_split
# Plot Import
import matplotlib.pyplot as plt

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




# ============= NORMALIZATION =============

def norm_sig( ecg_set ):
    ecg_set_normed = ecg_set
    
    for i, sig in enumerate(ecg_set):
        sig = norm_global_prime(sig)
        ecg_set_normed[i] = sig
    
    return ecg_set_normed




def norm_global( x ):
    """
    Normalized from [0,1] [min, max]
    """
    x_prime = (x - x.min()) / (x.max() - x.min() )
    return x_prime




def norm_global_prime( x ):
    """
    Normalized from [-1,1] [min, max] (uses norm_global(x) )
    """
    x = norm_global(x)
    x_prime = (2*x)-1
    return x_prime

# ============= 

def norm_ecg_subsets( ecg_denoised_flat, ecg_clean_flat ):
    diff = ecg_clean_flat[0] - ecg_denoised_flat[0]
    adjusted_ecg_flat = ecg_denoised_flat + diff
    return adjusted_ecg_flat



# ================ DATA MANIPULATION ================

def split_ecg_segments(ecg_data, start_minute=5):
    """
    WORKS WITH 360 SAMP FREQUENCY ONLY
    Returns 2 Numpy arrays consisting of noised segments and cleaned segments.
    Returned segments are based on the noise introduced by the NST script wherein 2 minutes of
    noised signals are followed by 2 minutes of clean signals (then repeat steps alternating).
    The resulting Numpy arrays are reshaped already in here so that it can be used directly with
    Autoencoder.model.fit(x).

    Returns:

        noised_seg (Numpy Array):
            Contiguous noised segments in the input ECG signal by NST
        
        clean_seg (Numpy Array):
            Contiguous clean segments in the input ECG untouched by NST

    """
    # How many usable 2-minute interval chunks are usable (rounded down) (Basically available sample points starting from the 5th-minute mark)
    # ISSUE: THE '300' USED IN HERE REFLECTS WHEN SAMPLING FREQUENCY IS 360 (THUS 1 SECOND) BUT DOES NOT PROPERLY REFLECT IF 
    # SAMPLING FREQUENCY IS 1024 WHERE 300 DOES NOT ANYMORE REFLECT THE 5TH MINUTE AS 1024 IS IS AROUND 2.88 SECONDS
    usable_chunks = int(((ecg_data.shape[0] - 300) / 120))
    # chunk size for the noised and clean segments
    segment_size = int( (usable_chunks/2) * 120 )
    # According to MIT-BIH, two minute intervals of noised and clean data will be generated by the NST script. This function assigns a 
    # contiguous version of those intervals in the variables below that will be returned
    noised_seg = np.zeros( shape=(segment_size, ecg_data.shape[1]) )
    clean_seg = np.zeros( shape=(segment_size, ecg_data.shape[1]) )
    # counters
    ecg_chunk_start = 60 * start_minute
    noised_chunk_start = 0
    clean_chunk_start = 0
    
    for i in range(usable_chunks):
        if i % 2 == 0:
            noised_seg[ noised_chunk_start:(noised_chunk_start+120) ] = ecg_data[ ecg_chunk_start:ecg_chunk_start+120 ]
            noised_chunk_start = noised_chunk_start+120
        else:
            clean_seg[ clean_chunk_start:(clean_chunk_start+120) ] = ecg_data[ ecg_chunk_start:ecg_chunk_start+120 ]
            clean_chunk_start = clean_chunk_start+120
        ecg_chunk_start = ecg_chunk_start+120

    noised_seg = noised_seg.reshape( (noised_seg.shape[0], noised_seg.shape[1], 1) )
    clean_seg = clean_seg.reshape( (clean_seg.shape[0], clean_seg.shape[1], 1) )

    return noised_seg, clean_seg




def realign_starting(ecg_result, ecg_clean):
    ecg_result = ecg_result
    ecg_clean_start = ecg_clean[0].cpu().numpy()
    diff = ecg_clean[0].cpu().numpy() - ecg_result[0]
    print( f'Diff: {diff}' )
    print( f'{ecg_clean_start} - {ecg_clean[0]}' )
    ecg_result = ecg_result + diff
    print( f'{ecg_clean_start} - {ecg_clean[0]}' )
    return ecg_result




def concat_pt_full(model, ecg_noisy):
    # Can only handle up to 4000 of the entire data set (then restart)
    # Firt Part
    result = model.encoder( ecg_noisy[0:4000] )
    result = model.decoder( result )
    result = result.detach().cpu().numpy()
    print( f'Result size: {result.shape}')
    np.save('res_pt2', result) # ranges from 0:4000
    # Second Part
    result = model.encoder( ecg_noisy[4000:5544] )
    result = model.decoder( result )
    result = result.detach().cpu().numpy()
    print( f'Result size: {result.shape}')
    np.save('res_pt2', result) # ranges from 4001:5544

    pt1 = np.load('res_pt1.npy')
    pt2 = np.load('res_pt2.npy')

    pt_full = np.concatenate( (pt1, pt2) )
    pt_full.shape
    np.save('res_pt_full', pt_full)




def ecg_plot_flat(ecg_clean, ecg_noisy, ecg_test, length=1024, index=0):
    ind_start = index * length
    ind_end = ind_start + length

    plt.figure( figsize=(20,5) )
    plt.plot( ecg_noisy[ind_start:ind_end], c='red', label='/original data' )
    plt.plot( ecg_clean[ind_start:ind_end], c='green', label='original data' )
    plt.plot( ecg_test[ind_start:ind_end], c='blue', label='Test Data' )
    plt.title(label='Sample')
    plt.legend()