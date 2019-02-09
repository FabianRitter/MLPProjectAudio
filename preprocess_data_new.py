import numpy as np
from librosa.feature import melspectrogram
import pandas as pd
import soundfile
import os


#CHANGE PATHS
path_to_metadata = '../../real_data/FSDnoisy18k.meta/train.csv'
base_path = '../../real_data/FSDnoisy18k.audio_train'

df_train = pd.read_csv(path_to_metadata)
fname = df_train['fname'].values

number_of_frames = 88200#88200 # they use 88200 
n_mels = 96
fmax = 22050
fmin = 0
fs= 44100
hop_length_samples = 882
n_fft = 2048
normalize_audio = True
patch_hop = 50
patch_len = 100
spectrogram_type = 'power'
win_length_samples = 1764


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
    
def convert2mel(audio,base_path):
    """
    Convert raw audio to mel spectrogram
    """
    path = os.path.join(base_path, audio)
    data, source_fs = soundfile.read(file=path)
    data = data.T
    # Resample if the source_fs is different from expected
    if fs != source_fs:
        data = librosa.core.resample(data, source_fs, params_extract.get('fs'))
        print('Resampling to %d: %s' % (params_extract.get('fs'), file_path))
    ### extracted from Eduardo Fonseca Code, it seems there are 3 audio corrupted so we need to check length
    
    if len(data) > 0 :
        data = normalize_amplitude(data)
    else: 
        ###### tenemos que ver como borrar estos audios! 
        data = np.ones((number_of_frames, 1))
        print('File corrupted. Could not open: %s' % path)
    #data = np.reshape(data, [-1, 1])        
    #data = data.astype(float)
    
    ## shor time processing of audio #####
    
    
    
    
    
    mels = melspectrogram(y=data, sr=44100,
                            n_fft=2048, hop_length=hop_length_samples,
                            power=2.0, n_mels=n_mels,fmax=fmax) 
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

#all_inputs = np.zeros([len(fname),n_mels,number_of_frames])
all_inputs= []
all_targets = np.zeros(len(fname))
all_manually = np.zeros(len(fname))
all_noisy_small = np.zeros(len(fname))
all_dict = {}
#all_inputs = [convert2mel(audio,base_path) for audio in fname]                      
all_inputs.append([convert2mel(audio,base_path) for audio in fname])
np.asarray(all_inputs, dtype=np.float32)
all_dict['inputs'] = all_inputs
all_dict['targets'] =  df_train['label']
all_dict['manually_verified'] = df_train['manually_verified']
all_dict['noisy_small'] = df_train['noisy_small']

np.savez('processed_data-train.npz',**all_dict)