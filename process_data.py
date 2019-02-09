import numpy as np
from librosa.feature import melspectrogram
import pandas as pd
from scipy.io import wavfile
import os
import time
from tqdm import tqdm, trange
import argparse
import yaml



####### ARGUMENTS (JUST THE PATH FOR NOW) #######

parser = argparse.ArgumentParser(description='Code for DCASE Challenge task 2.')

parser.add_argument('-p','--params',dest='params_preprocessing',action='store',
                        required=False,type=str)

args = parser.parse_args()

if args.params_preprocessing:
    params = yaml.load(open(args.params_preprocessing))
    path_to_metadata = params['dataset_path']
else:
#CHANGE PATHS
    path_to_metadata = '../../real_data/FSDnoisy18k.meta/train.csv'
    base_path = '../../real_data/FSDnoisy18k.audio_train'

df_train = pd.read_csv(path_to_metadata)
fname = df_train['fname'].values

number_of_frames = 100
n_mels = 96

all_inputs = np.zeros([len(fname),number_of_frames*n_mels])
all_targets = np.zeros(len(fname))
all_manually = np.zeros(len(fname))
all_noisy_small = np.zeros(len(fname))
all_dict = {}

start = time.time()

for ii,audio in enumerate(fname):
    path = os.path.join(base_path, audio)
    fs, data = wavfile.read(path)
    data = data.astype(float)
    mels = melspectrogram(y=data, sr=44100,
                            n_fft=2048, hop_length=512,
                            power=2.0, n_mels=96,fmax=16000)


    mel_normalized = normalize_mel_histogram(mels.T,number_of_frames)
    mel_norm_flat = mel_normalized.flatten()
    all_inputs[ii] = mel_norm_flat
    all_targets = df_train.iloc[ii]['label']
    all_manually = df_train.iloc[ii]['manually_verified']
    all_noisy_small = df_train.iloc[ii]['noisy_small']
end = time.time()

print('Time elapsef for pre processing: %7.2f hours') % ((end-start)/ 3600.0)

all_dict['inputs'] = all_inputs
all_dict['targets'] = all_targets
all_dict['manually_verified'] = all_manually
all_dict['noisy_small'] = all_noisy_small

np.savez('processed_data.npz',all_dict)














############FUNCTIONS#################
# make all data same size just taking 2 seconds of audio.

def normalize_mel_histogram(mel_hist, number_of_frames=100):
    """
    Return a normalized mel histogram

    mel_hist:
    number_of_frames: number of frames to be normalized
    """
    if mel_hist.shape[0] > number_of_frames:
        return np.delete(mel_hist,
                         np.arange(number_of_frames, mel_hist.shape[0]),
                         axis=0)

    elif mel_hist.shape[0] < number_of_frames:
        mul = int(round(number_of_frames / mel_hist.shape[0], 0)) + 1
        repeated_matrix = np.tile(mel_hist.T, mul).T

        if repeated_matrix.shape[0] > number_of_frames:
            return np.delete(repeated_matrix,
                             np.arange(number_of_frames, repeated_matrix.shape[0]),
                             axis=0)
        return repeated_matrix

    else:
        return mel_hist
