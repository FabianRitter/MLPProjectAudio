import numpy as np
import librosa
from librosa.feature import melspectrogram
import pandas as pd
import soundfile
import argparse
import yaml
import os
import h5py
from utils.preprocessing import convert2mel, normalize_amplitude, windowing



####### ARGUMENTS (JUST THE PATH FOR NOW) #######
#### we have to normalize the amplitude and also the spectrum!! because of the microphones!!

"""
Train entries: 17585
New train entries with some space for validationL: 17310 -> 271 batches
val set: 275 -> 5 batches
Test entries: 947 -> 15 batches
Obs: we will start working with batches using chunking
"""
parser = argparse.ArgumentParser(description='Code for DCASE Challenge task 2.')
parser.add_argument('-p','--params',dest='params_preprocessing',action='store',
                        required=False,type=str)
parser.add_argument('-e','--expnum',dest='experiment_number', action='store',
                        required=False,type=int)
parser.add_argument('-t','--testing',dest='testing', action='store', required=False, type=str,default=False)
args = parser.parse_args()


if args.params_preprocessing:
    params = yaml.load(open(args.params_preprocessing))
    params_ctrl = params['ctrl']
    path_to_metadata = params_ctrl.get('dataset_path')
    if args.testing:
        type_training = args.testing
    else:
        type_training = params_ctrl.get('type_training')
    chunk = int(params_ctrl.get('chunk_size'))
    path_to_metadata = '/disk/scratch/s1870525/datasets/FSDnoisy18k.meta/' + type_training + "_set.csv"
    base_path= '/disk/scratch/s1870525/datasets/' + type_training
    if type_training == 'valid':
    	base_path = '/disk/scratch/s1870525/datasets/train'

    hdf5_name = "processed_data_" + type_training  +  ".hdf5"
    print('base path is', base_path)

if args.experiment_number:
    experiment_number= int(args.experiment_number)
else:
    experiment_number = False

df_train = pd.read_csv(path_to_metadata)
fname = df_train['fname'].values

n_mels = 96
fs= 32000 # we will make downsampling to save some data!!44100
n_fft = 512
windows_size_s = 35 # 30 milisecons windowing (to have more context)
windows_size_f = (windows_size_s * fs ) // 1000  # int division # 960 samples
hop_length_samples = int(windows_size_f // 2) ## 480 samples
audio_duration_s = 1.45  # 2 seconds
audio_duration = audio_duration_s * 1000
number_of_frames = fs * audio_duration # deprecated, use short audio in database already
fmax = int(fs / 2)
fmin = 0
normalize_audio = True
spectrogram_type = 'power'
maximum_mel = 0


if experiment_number and type_training == "train":
    if experiment_number == 271:
        fname = fname[chunk * (experiment_number-1): len(df_train['fname'].values) ]
    else:
        fname = fname[chunk * (experiment_number-1): chunk * experiment_number]
    print("the length of fname is", len(fname))
elif type_training == "test":
    if experiment_number == 15:
        fname = fname[chunk * (experiment_number-1):len(df_train['fname'].values)]
    else:
        fname = fname[chunk * (experiment_number-1): chunk * experiment_number]
    print('using {0} files for testing data'.format(len(fname)))
else:
    if experiment_number == 5:
        fname = fname[chunk * (experiment_number-1): len(df_train['fname'].values)]
    else:
        fname = fname[chunk * (experiment_number-1): chunk * experiment_number]


if experiment_number == 1:
    hdf5_store = h5py.File(hdf5_name, "w")
    #all_inputs = hdf5_store.create_dataset("all_inputs-batch-" + experiment_number, (len(df_train['fname'].values),n_mels*number_of_frames), compression="gzip")
    all_inputs = hdf5_store.create_dataset("all_inputs" , (len(df_train['fname'].values),1, n_mels ,audio_duration // hop_length_samples), chunks= (64 , 1 ,n_mels,audio_duration // hop_length_samples)   ,compression="gzip")
    dt = h5py.special_dtype(vlen=str)
    targets = hdf5_store.create_dataset("targets", data = df_train['label'].values, dtype=dt ,compression="gzip")
    data_processed = [convert2mel(audio,base_path,fs, n_fft,fmax,n_mels,hop_length_samples, windows_size_f) for ii,audio in enumerate(fname)]
    all_inputs[chunk * (experiment_number-1) :chunk * experiment_number , 1 ] = data_processed

    if type_training == 'train' or type_training == 'val':
        manually_verified = hdf5_store.create_dataset("manually_verified", dtype='i1' ,data = df_train['manually_verified'].values, compression="gzip")
        noisy_small =  hdf5_store.create_dataset("noisy_small", dtype='i1' ,data = df_train['noisy_small'].values, compression="gzip")

else:
    hdf5_store = h5py.File(hdf5_name, "a")
    data_processed = [convert2mel(audio,base_path,fs, n_fft,fmax,n_mels,hop_length_samples, windows_size_f) for ii,audio in enumerate(fname)]

    hdf5_store['all_inputs'][chunk * (experiment_number-1) :chunk * experiment_number,1] = data_processed
print("maximum_mel is", maximum_mel)
print("saving data for experiment" , experiment_number)


hdf5_store.close()
