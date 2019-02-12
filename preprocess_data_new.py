import numpy as np
import librosa
from librosa.feature import melspectrogram
import pandas as pd
import soundfile
import argparse
import yaml
import os
import h5py




####### ARGUMENTS (JUST THE PATH FOR NOW) #######

parser = argparse.ArgumentParser(description='Code for DCASE Challenge task 2.')

parser.add_argument('-p','--params',dest='params_preprocessing',action='store',
                        required=False,type=str)

parser.add_argument('-e','--expnum',dest='experiment_number', action='store',
                        required=False,type=int)


args = parser.parse_args()


if args.params_preprocessing:
    params = yaml.load(open(args.params_preprocessing))
    params_ctrl = params['ctrl']
    path_to_metadata = params_ctrl.get('dataset_path')
    type_training = params_ctrl.get('type_training')
    path_to_metadata = '../../real_data/FSDnoisy18k.meta/' + type_training + ".csv"
    base_path = '../../real_data/FSDnoisy18k.audio_' + type_training
    hdf5_name = "processed_data_" + type_training  +  ".hdf5"
    print('base path is', base_path)
else:
#CHANGE PATHS
    path_to_metadata = '../../real_data/FSDnoisy18k.meta/train.csv'
    base_path = '../../real_data/FSDnoisy18k.audio_train'

if args.experiment_number:
    experiment_number= args.experiment_number
else:
    experiment_number = False



df_train = pd.read_csv(path_to_metadata)
fname = df_train['fname'].values



n_mels = 64
win_length_samples = 512
fs= 16000 # we will make downsampling to save some data!!44100
number_of_frames = fs * 2 # two seconds of data-88200#88200 # they use 88200 
hop_length_samples = int(win_length_samples / 2)
fmax = int(fs / 2)
fmin = 0
n_fft = 1024
normalize_audio = True
patch_hop = 50
patch_len = 100
spectrogram_type = 'power'


if experiment_number and type_training == "train":
    if experiment_number == 176:
        fname = fname[100 * (experiment_number-1):len(fname)]
    fname = fname[100 * (experiment_number-1): 100 * experiment_number]
    print("the length of fname is", len(fname))
else:
    if experiment_number == 10:
        fname = fname[100 * (experiment_number-1):len(fname)]
    fname = fname[100 * (experiment_number-1): 100 * experiment_number]
    print('using {0} files for testing data'.format(len(fname)))


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
    
def convert2mel(audio,base_path,fs,fmax,n_mels,number_of_frames, counter):
    """
    Convert raw audio to mel spectrogram
    """
        
    path = os.path.join(base_path, audio)
    data, source_fs = soundfile.read(file=path)
    data = data.T
    # Resample if the source_fs is different from expected
    if fs != source_fs:
        data = librosa.core.resample(data, source_fs,fs)
    ### extracted from Eduardo Fonseca Code, it seems there are 3 audio corrupted so we need to check length
    
    if len(data) > 0 :
        data = normalize_amplitude(data)
    else: 
        ###### tenemos que ver como borrar estos audios! 
        data = np.ones((number_of_frames, 1))
        print('File corrupted. Could not open: %s' % path)
    
    mels = melspectrogram(y=data, sr=fs,
                            n_fft=2048, hop_length=hop_length_samples,
                            power=2, n_mels=n_mels,fmax=fmax) 
    mel_normalized = normalize_mel_histogram(mels.T,number_of_frames)
    mel_norm_flat = mel_normalized.flatten()
    return mel_norm_flat


##### Amplitude Normalization of audios #########

def normalize_amplitude(y, tolerance=0.005):

    mean_value = np.mean(y)
    y -= mean_value

    max_value = max(abs(y)) + tolerance
    return y / max_value

processes = []

if experiment_number == 1:
    hdf5_store = h5py.File(hdf5_name, "w")
    #all_inputs = hdf5_store.create_dataset("all_inputs-batch-" + experiment_number, (len(df_train['fname'].values),n_mels*number_of_frames), compression="gzip")
    all_inputs = hdf5_store.create_dataset("all_inputs" , (len(df_train['fname'].values),n_mels*number_of_frames), compression="gzip")  
    dt = h5py.special_dtype(vlen=str)
    targets = hdf5_store.create_dataset("targets", data = df_train['label'].values, dtype=dt ,compression="gzip")
    if type_training == 'train':
        manually_verified = hdf5_store.create_dataset("manually_verified", data = df_train['manually_verified'].values, compression="gzip")
        noisy_small =  hdf5_store.create_dataset("noisy_small", data = df_train['noisy_small'].values, compression="gzip")
else:
    hdf5_store = h5py.File(hdf5_name, "a")


all_inputs = [convert2mel(audio,base_path,fs,fmax,n_mels,number_of_frames,ii) for ii,audio in enumerate(fname)]                      

print("saving data for experiment" , experiment_number)


hdf5_store.close()



