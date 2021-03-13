# Numpy
import numpy as np
# math
import math
import time

def compare_qrs_detect(sig_denoised, sig_noisy):
    sig_denoised = sig_denoised.numpy()
    pass

# ===== USE THIS INSTEAD =====
def get_snr_imp(ecg_clean, ecg_noisy, ecg_denoised):
    # Compute for the numerator
    def get_upper( ecg_clean ):
        upper = 0
        for i in range( len(ecg_clean) ):
            upper = upper + ( ecg_clean[i]**2 )
        return upper
    # Compute for denominator
    def get_lower( ecg_clean, ecg_ref ):
        lower = 0
        # Compute for denominator
        for i in range( len(ecg_clean) ):
            lower = lower + ( (ecg_ref[i] - ecg_clean[i])**2 )
        return lower
    upper = get_upper( ecg_clean )
    lower_in = get_lower( ecg_clean, ecg_noisy )
    lower_out = get_lower( ecg_clean, ecg_denoised )
    snr_in = 10 * ( math.log10( upper/lower_in) )
    snr_out = 10 * ( math.log10( upper/lower_out) )
    snr_imp = snr_out - snr_in
    print( f'SNR_IN: {snr_in}')
    print( f'SNR_out: {snr_out} ')
    print( f'SNR Improvement: {snr_imp}')
    return snr_imp

def get_rmse( ecg_clean, ecg_ref ):
    result = 0
    # Compute for the summation in formula
    def get_sumt_res( ecg_clean, ecg_ref ):
        sumt_res = 0
        # Compute for denominator
        for i in range( len(ecg_clean) ):
            sumt_res = sumt_res + ( (ecg_clean[i] - ecg_ref[i])**2 )
        return sumt_res
    
    result = math.sqrt( (1/len(ecg_clean)) * (get_sumt_res(ecg_clean, ecg_ref)) )
    print( f'RMSE: {result}' )
    return result

def get_prd(ecg_clean, ecg_ref):
    result = 0
    def get_numerator( ecg_clean, ecg_ref ):
        res = 0
        # Compute for denominator
        for i in range( len(ecg_clean) ):
            res = res + ( (ecg_clean[i] - ecg_ref[i])**2 )
        return res
    def get_denominator( ecg_clean):
        res = 0
        for i in range( len(ecg_clean) ):
            res = res + ( ecg_clean[i]**2 )
        return res
    result = math.sqrt( ((get_numerator(ecg_clean, ecg_ref)/get_denominator(ecg_clean)) * 100) )
    print( f'PRD: {result}' )
    return result

def get_eval_metrics(ecg_clean, ecg_noisy, ecg_denoised, verb=True):
    def compute_x_sqrd( ecg_clean ):
        res = 0
        for i in range( len(ecg_clean) ):
            res = res + ( ecg_clean[i]**2 )
        return res
    def compute_snr_denominator( ecg_clean, ecg_ref):
        lower = 0
        # Compute for denominator
        for i in range( len(ecg_clean) ):
            lower = lower + ( (ecg_ref[i] - ecg_clean[i])**2 )
        return lower
    def compute_prd_num( ecg_clean, ecg_ref ):
        res = 0
        # Compute for denominator
        for i in range( len(ecg_clean) ):
            res = res + ( (ecg_clean[i] - ecg_ref[i])**2 )
        return res
    # Timer start
    elapsed_start = time.time()
    # result[SNRimp, RMSE, PRD]
    result = [0, 0, 0]
    # populate constants when computing
    x_sqrd = compute_x_sqrd(ecg_clean) # used in all 3 eval metrics
    recon_diff = compute_prd_num(ecg_clean, ecg_denoised) # used in PRD and RMSE
    snr_in_denom = compute_snr_denominator(ecg_clean, ecg_noisy)
    snr_out_denom = compute_snr_denominator(ecg_clean, ecg_denoised)
    # Compute SNR in and out
    snr_in_res = 10 * ( math.log10( x_sqrd/snr_in_denom) )
    snr_out_res = 10 * ( math.log10( x_sqrd/snr_out_denom) )
    # Final eval results
    snr_imp = snr_out_res - snr_in_res
    rmse = math.sqrt( (1/len(ecg_clean)) * ( recon_diff ) )
    prd = math.sqrt( ((recon_diff/x_sqrd) * 100) )
    result = [snr_imp, rmse, prd]
    if verb:
        print( f'SNR Improvement: {snr_imp}')
        print( f'RMSE: {rmse}')
        print( f'PRD: {prd}')
    # Show elapsed time
    elapsed_end = time.time()
    elapsed_total = elapsed_end-elapsed_start
    elapsed_mins = int(elapsed_total/60)
    elapsed_secs = int(elapsed_total - (60 * elapsed_mins))
    print(f'Elapsed time: {elapsed_total:.2f}, (in mins: {elapsed_mins}:{elapsed_secs})')
    
    return result