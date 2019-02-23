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
    #maximum_mel = int(params_ctrl.get('maximum_mel'))
    #minimum_mel = int(params_ctrl.get('minimum_mel'))
    path_to_metadata = '../../real_data/FSDnoisy18k.meta/' + type_training + "_set.csv"
    base_path = '../../real_data/FSDnoisy18k.audio_' + type_training
    if type_training == 'val':
    	base_path = '../../real_data/FSDnoisy18k.audio_train'

    hdf5_name = "processed_data_" + type_training  +  ".hdf5"
    print('base path is', base_path)
else:
#CHANGE PATHS
    path_to_metadata = '../../real_data/FSDnoisy18k.meta/train.csv'
    base_path = '../../real_data/FSDnoisy18k.audio_train'

if args.experiment_number:
    experiment_number= int(args.experiment_number)
else:
    experiment_number = False




df_train = pd.read_csv(path_to_metadata)

fname = df_train['fname'].values
 


n_mels = 64
n_fft = 1024
fs= 16000 # we will make downsampling to save some data!!44100
number_of_frames = fs * 2 # two seconds of data-88200#88200 # they use 88200 
hop_length_samples = int(n_fft / 2)
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
	if experiment_number = 5:
		fname = fname[chunk * (experiment_number-1): len(df_train['fname'].values) ]
	else:
        fname = fname[chunk * (experiment_number-1): chunk * experiment_number]




def normalize_mel_histogram(mel_hist, number_of_frames=32000):
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
    
def convert2mel(audio,base_path,fs, n_fft,fmax,n_mels,number_of_frames):
    """
    Convert raw audio to mel spectrogram
    """
    global maximum_mel

    path = os.path.join(base_path, audio)
    data, source_fs = soundfile.read(file=path)
    data = data.T
    # Resample if the source_fs is different from expected
    if fs != source_fs:
        data = librosa.core.resample(data, source_fs,fs)
    ### extracted from Eduardo Fonseca Code, it seems there are 3 audio corrupted so we need to check length
    data = normalize_amplitude(data)
    mels = melspectrogram(y= data , sr=fs,
                            n_fft=2048, hop_length=hop_length_samples,
                            power=2, n_mels=n_mels,fmax=fmax) 
    mel_normalized = normalize_mel_histogram(mels.T,number_of_frames)
    mel_normalized = (mel_normalized -  np.mean(mel_normalized, axis =0)) / np.amax(mel_normalized) 
    if mel_normalized.max() > maximum_mel:
        maximum_mel = mel_normalized.max()
        
    return mel_normalized.flatten()



##### Amplitude Normalization of audios #########

def normalize_amplitude(y, tolerance=0.005):

    mean_value = np.mean(y)
    y -= mean_value

    max_value = max(abs(y)) + tolerance
    return y / max_value

processes = []

#####

def imprimir(ii,audio,base_path,fs, n_fft,fmax,n_mels,number_of_frames):
    mel = convert2mel(audio,base_path,fs, n_fft,fmax,n_mels,number_of_frames)
    if ii == 0:
        print(mel)
    return mel

####

if experiment_number == 1:
    hdf5_store = h5py.File(hdf5_name, "w")
    #all_inputs = hdf5_store.create_dataset("all_inputs-batch-" + experiment_number, (len(df_train['fname'].values),n_mels*number_of_frames), compression="gzip")
    all_inputs = hdf5_store.create_dataset("all_inputs" , (len(df_train['fname'].values),n_mels*number_of_frames), chunks= (64, n_mels * number_of_frames)   ,compression="gzip")  
    dt = h5py.special_dtype(vlen=str)
    targets = hdf5_store.create_dataset("targets", data = df_train['label'].values, dtype=dt ,compression="gzip")
    data_processed = [convert2mel(audio,base_path,fs, n_fft,fmax,n_mels,number_of_frames) for ii,audio in enumerate(fname)]
    all_inputs[chunk * (experiment_number-1) :chunk * experiment_number] = data_processed 
    if type_training == 'train':
        manually_verified = hdf5_store.create_dataset("manually_verified", dtype='i1' ,data = df_train['manually_verified'].values, compression="gzip")
        noisy_small =  hdf5_store.create_dataset("noisy_small", dtype='i1' ,data = df_train['noisy_small'].values, compression="gzip")

else:
    hdf5_store = h5py.File(hdf5_name, "a")
    data_processed = [convert2mel(audio,base_path,fs, n_fft,fmax,n_mels,number_of_frames) for ii,audio in enumerate(fname)]

    hdf5_store['all_inputs'][chunk * (experiment_number-1) :chunk * experiment_number] = data_processed


print("maximum_mel of batch", maximum_mel)
print("saving data for experiment" , experiment_number)




hdf5_store.close()

