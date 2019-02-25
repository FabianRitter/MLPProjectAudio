import numpy as np
import librosa
from librosa.feature import melspectrogram
import pandas as pd
import soundfile
import os
#import h5py


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
    
def normalize_amplitude(y, tolerance=0.005):

    mean_value = np.mean(y)
    y -= mean_value

    max_value = max(abs(y)) + tolerance
    return y / max_value
    
def convert2mel(audio,fs=16000, n_fft=2048,base_path="/home/jordi/mlp_audio/MLPProjectAudio/MLP_CW2/data/audio",
                fmax=512,n_mels=2,number_of_frames=2):
    """
    Convert raw audio to mel spectrogram
    """
    #global maximum_mel
    #print(maximum_mel)
    hop_length_samples = int(n_fft / 2)

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
    #if mel_normalized.max() > maximum_mel:
    #    maximum_mel = mel_normalized.max()
    return mel_normalized.flatten()

