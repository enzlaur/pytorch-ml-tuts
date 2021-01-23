# ==== IMPORTS ====
# ---- PYTORCH  ----
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, random_split
# ---- OTHER IMPORTS ----
import wfdb
import peakutils
import numpy as np
import pandas as pd
import time # to be used by timer
# ---- SKLEARN ----
from sklearn import preprocessing # for normalizing data
from sklearn.model_selection import train_test_split # For test data splitting



def train_model(model, epochs, ecg_noisy, ecg_clean, train_pct=0.8):
    
    # GET THE APPROPRIATE SIZES FOR TRAINING AND VALIDATION
    def get_train_valid_sizes( ecg_clean ):
        # valid_pct = 1 - train_pct # auto compute for the validation
        # Compute for the sizes of training and validation
        total_size = ecg_clean.shape[0]
        train_size = int( (total_size) * train_pct )
        valid_size = total_size - train_size
        
        print( f'train_size[{train_size}] + valid_size[{valid_size}]= {total_size}')
        if( (train_size+valid_size) == total_size):
            print(f'same size')
        else:
            print(f'not same size')
        return train_size, valid_size, total_size

    # CREATE THE TRAINDEX (TRAINING INDEX SETS)
    def create_index_loaders( ecg_clean ):
        # Get train and validation set sizes
        train_size, valid_size, total_size = get_train_valid_sizes(ecg_clean)    
        # Well instead of having to randomize the data itself, why not the numbers used to index
        index_set = np.arange(0, total_size)

        # split the indexes used for training and validation
        train_indexset, val_indexset = random_split( index_set, (train_size, valid_size))

        # Create DataLoaders for the corresponding train and val sets
        traindex_loader = DataLoader( train_indexset, shuffle=True, batch_size=1 )
        validex_loader = DataLoader( val_indexset, shuffle=True, batch_size=1 )
        return traindex_loader, validex_loader

    elapsed_start = time.time() # used to compute for elapsed time for each epoch and entire training
    train_loss = []
    criterion = nn.MSELoss().cuda()
    optimizer = torch.optim.Adam( model.parameters(), lr=1e-3 )
    # Prep the traindex, validex
    traindex_loader, validex_loader = create_index_loaders( ecg_clean )

    for epoch in range(epochs):
        # Running loss computed at the end
        running_loss = 0.0        
        # start timer
        epoch_start = time.time()
        # Loop through the entire dataset
        for i, data_index in enumerate(traindex_loader):
            # Load the noise sample
            index = data_index.numpy()[0]
            noise_samp = ecg_noisy[index]
            clean_samp = ecg_clean[index]
            noise_samp = noise_samp.view( 1, noise_samp.shape[0], noise_samp.shape[1])
            clean_samp = clean_samp.view( 1, clean_samp.shape[0], clean_samp.shape[1])
            # Convert x_samps to tensors and in cuda
            optimizer.zero_grad()
            # one_sig = noise_sig
            # one_sig = noise_sig.view( noise_sig.shape[1], noise_sig.shape[0], 1)
            
            x_prime = model( noise_samp )
            
            loss = criterion( x_prime, clean_samp) # or loss function
            
            # Backpropagation
            loss.backward()
            
            optimizer.step()
            running_loss += loss.item()
        
            loss = running_loss / len(traindex_loader)
        
            train_loss.append(loss)

        # ===== Epoch timer =====
        epoch_end = time.time()
        time_total = epoch_end - epoch_start
        print( f"Epoch {epoch+1} of {epochs} || time: {time_total:.2f} || loss = {loss}")
        # ===== Total training elapsed time =====
    elapsed_end = time.time()
    elapsed_total = elapsed_end-elapsed_start
    elapsed_mins = int(elapsed_total/60)
    elapsed_secs = int(elapsed_total - (60 * elapsed_mins))
    print(f'Elapsed time: {elapsed_total:.2f}, (in mins: {elapsed_mins}:{elapsed_secs})')
    print(f'Validation dataset has not been used: Available validex set size = {len(validex_loader)}')
    return train_loss