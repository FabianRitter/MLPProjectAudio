import numpy as np
import librosa
from librosa.feature import melspectrogram
from librosa.core import stft
import soundfile
import os


"""
Functions for the pre processing

"""

def convert2mel(audio,base_path,fs, n_fft,fmax,n_mels,hop_length_samples, window_lenght):
    """
    Convert raw audio to mel spectrogram
    """
    global maximum_mel

    path = os.path.join(base_path, audio)
    data, _ = librosa.core.load(path, sr=fs, res_type="kaiser_best")

    # Resample if the source_fs is different from expected
    #if fs != source_fs:
    #    data = librosa.core.resample(data, source_fs,fs)
    ### extracted from Eduardo Fonseca Code, it seems there are 3 audio corrupted so we need to check length
    data = normalize_amplitude(data)

    powSpectrum = np.abs(stft(data,n_fft,hop_length = hop_length_samples, win_length = window_lenght, window = 'hamming', center=True, pad_mode='reflect'))**2

    mels = melspectrogram(y= None,n_fft=n_fft ,sr=fs ,S= powSpectrum, hop_length= hop_length_samples ,n_mels=n_mels,fmax=fmax , fmin = 0.0).T
    #mel_normalized = (mels -  np.mean(mels, axis =0)) / np.amax(mels)
    #if mel_normalized.max() > maximum_mel:
    #    maximum_mel = mel_normalized.max()
    #to make it to db... we can add a mfcc optional
    mels = librosa.core.power_to_db(mels, ref=np.min(mels))
    mels = mels / np.max(mels)

    return mels.T

##### Amplitude Normalization of audios #########

def normalize_amplitude(y, tolerance=0.005):

    mean_value = np.mean(y)
    y -= mean_value

    max_value = max(abs(y)) + tolerance
    return y / max_value

processes = []

#####

def windowing(win_leng_frames, sym=False):
    return scipy.signal.hamming(win_leng_frames, sym=False)



def imprimir(ii,audio,base_path,fs, n_fft,fmax,n_mels,number_of_frames):
    mel = convert2mel(audio,base_path,fs, n_fft,fmax,n_mels,number_of_frames)
    if ii == 0:
        print(mel)
    return mel

####
