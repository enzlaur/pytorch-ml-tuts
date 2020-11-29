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

# ============= NORMALIZATION =============
def normalize(sigs):
    # subtract by avg value / max of absolute value
    #  e.g.: [sigs](-5.635) -  [avg](-8.70) / 9.135 = -1.569
    avg = get_avg(sigs)
    max_val = get_max_abs(sigs)
    normed = np.zeros(shape=(sigs.shape[0], sigs.shape[1]))

    for i in range(sigs.shape[0]):
        for j in range(sigs.shape[1]):
            normed[i][j] = (sigs[i][j] - avg) / max_val

    return normed

def per_chunk_max_val(sig):
    max_val = 0
    for i in range(sig.shape[0]):
            if abs(sig[i]) > max_val:
                max_val = abs(sig[i])
    
    return max_val

def per_chunk_avg(sig):
    sums = 0
    avg = sig.shape[0]
    for i in range(len(sig)):
        sums = sums + sig[i]
    avg = sums / avg

    return avg

def per_chunk_norm(sig):
    avg = per_chunk_avg(sig)
    max_abs_val = per_chunk_max_val(sig)
    # print("average " + str(avg) + "||" + "max_abs " + str(max_val)) 
    normed = np.zeros(shape=(sig.shape[0]))
    for i in range( len(sig) ):
        normed[i] = (sig[i] - avg) / max_abs_val
    
    return normed

def normalize_by_chunks(sigs):
    """
    Normalizes ecg signal based on sampling frequency chunk (e.g. denominator of the mean is the sampling frequency and not the entire ecg signal)

    Returns::
        sigs (nd array): normalized ECG signal

    Paramter::
        sigs (nd array): original ECG signal
    """
    for i in range(sigs.shape[0]):
        sigs[i] = per_chunk_norm(sigs[i])

    return sigs

def get_samples(data_name, samp_freq=360, channel=0, norm_type=''):
    """
    Returns:

        signal (list): 2D array-like list [x][y] (not numpy) where first index is chunks and the 2nd index contains sample points from a single iteration determined by your cycle_len.
        The selected channel will automatically be V1

    Parameters::

        data_name (str): File name to be accessed (from MIT-BIH/Physionet) (e.g. ``100``, ``124e06``)
        samp_freq (int): Sample frequecy determined by you source 
        channel (int): Channel from the signal to use
        norm_type (str): Default is '', options are ``chunks`` and ``all``. Chunks option is divided by sampling frequency, all option is dividing by average of all datapoints in the entire signal

    Example:

        get_samples(data_name='100', samp_freq=360)
    """
    # load the signal via wfdb
    signal = load_signal(data_name=data_name, channel=channel)
    
    chunks = int(len(signal) / samp_freq)
    
    signal_list = np.zeros(shape=(chunks, samp_freq))
    
    sig_counter = 0
    for i in range(chunks):
        for j in range(samp_freq):
            signal_list[i][j] = signal[sig_counter]
            sig_counter = sig_counter + 1

    # Perform normalization.
    # normalization per chunk (normalizes data points as divided by sampling frequency)
    if str.lower(norm_type) == 'chunks' or str.lower(norm_type) == 'chunk':
        signal_list = normalize_by_chunks(signal_list)
    if str.lower(norm_type) == 'all':
        signal_list = normalize(signal_list)
    if str.lower(norm_type) == 'none':
        pass

    return signal_list

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

# Default GET_ECG method to use. Don't use others anymore huhu
def get_ecg_with_split(data_name, samp_freq=360, norm_type='chunks', channel=1):
    """
    DEFAULT GET_ECG METHOD TO USE
    
    Use when samp_freq is not 360 (specifically the 1024 sampling frequency)

    Returns:
        noised_seg, clean_seg (Numpy Array):
            Wherein both arrays contains time-specific segments of noise and untouched ECG from the NST generator.
    """
    # load everything with a sampling frequency of 360 (easier to split)
    x = get_samples(data_name, samp_freq=360, channel=channel, norm_type='') # must not normalize in this part yet
    # Split into two (using 360 as sampling frequency as 360Hz == 1 second)
    noised_seg, clean_seg = split_ecg_segments(x)
    # Flatten the arrays noised_seg and clean_seg into 1D
    noised_seg = noised_seg.flatten()
    clean_seg = clean_seg.flatten()
    # Once flatten, return both ECG's according to their sampling frequency
    # Get the appropriate length for chunks first
    chunks_noised = int( len(noised_seg) / samp_freq )
    chunks_clean = int( len(clean_seg) / samp_freq )
    chunks = 0
    # Assures that the smaller chunk size is followed for consistency (esp during training). 
    if chunks_noised < chunks_clean:
        chunks = chunks_noised
    elif chunks_noised > chunks_clean:
        chunks = chunks_clean
    else:
        chunks = chunks_noised
    # create the np arrays to be returned
    noised_seg_new = np.zeros( (int(chunks/2), samp_freq) ) # why + 1, idk yet
    clean_seg_new = np.zeros( (int(chunks/2), samp_freq) )
    # Proceed to migrate into 2D Arrays
    for i in range( chunks - 1 ):
        if i % 2 == 0:
            noised_seg_new[ int(i/2) ] = noised_seg[ (samp_freq*i):((samp_freq*i)+samp_freq) ]
        else:
            clean_seg_new[ int(i/2 )] = clean_seg[ (samp_freq*i):((samp_freq*i)+samp_freq) ]
    
    # Perform normalization.
    # normalization per chunk (normalizes data points as divided by sampling frequency)
    if str.lower(norm_type) == 'chunks' or str.lower(norm_type) == 'chunk':
        noised_seg_new = normalize_by_chunks(noised_seg_new)
        clean_seg_new = normalize_by_chunks(clean_seg_new)
    if str.lower(norm_type) == 'all':
        noised_seg_new = normalize(noised_seg_new)
        clean_seg_new = normalize(clean_seg_new)
    if str.lower(norm_type) == 'none':
        pass
    # Reshape ready for training
    clean_seg_new = clean_seg_new.reshape( (clean_seg_new.shape[0], clean_seg_new.shape[1], 1) )
    noised_seg_new = noised_seg_new.reshape( (noised_seg_new.shape[0], noised_seg_new.shape[1], 1) )
    return noised_seg_new, clean_seg_new