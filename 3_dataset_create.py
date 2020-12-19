
from google.colab import drive
drive.mount('/content/drive')

import librosa
import os
import pandas as pd
from numpy import mean
import warnings;
warnings.filterwarnings('ignore');

folders_5s = {
              'pop_5s':'/content/drive/My Drive/ML_Project/New_Data/pop_test_5s',
              'rnb_5s':'/content/drive/My Drive/ML_Project/New_Data/rnb_test_5s', 
              'blues_5s':'/content/drive/My Drive/ML_Project/New_Data/blues_test_5s',
              'hiphop_5s':'/content/drive/My Drive/ML_Project/New_Data/hiphop_test_5s',
              'rock_5s':'/content/drive/My Drive/ML_Project/New_Data/rock_test_5s'
            }

folders_10s = {
              'pop_10s':'/content/drive/My Drive/ML_Project/New_Data/pop_test_10s',
              'rnb_10s':'/content/drive/My Drive/ML_Project/New_Data/rnb_test_10s', 
              'blues_10s':'/content/drive/My Drive/ML_Project/New_Data/blues_test_10s',
              'hiphop_10s':'/content/drive/My Drive/ML_Project/New_Data/hiphop_test_10s',
              'rock_10s':'/content/drive/My Drive/ML_Project/New_Data/rock_test_10s'
            }

folders_20s = {
              'pop_20s':'/content/drive/My Drive/ML_Project/New_Data/pop_test_20s',
              'rnb_20s':'/content/drive/My Drive/ML_Project/New_Data/rnb_test_20s', 
              'blues_20s':'/content/drive/My Drive/ML_Project/New_Data/blues_test_20s',
              'hiphop_20s':'/content/drive/My Drive/ML_Project/New_Data/hiphop_test_20s',
              'rock_20s':'/content/drive/My Drive/ML_Project/New_Data/rock_test_20s'
            }


label = {
          'pop_5s': 0, 'rnb_5s': 1, 'blues_5s': 2, 'hiphop_5s': 3, 'rock_5s': 4, 
          'pop_10s': 0, 'rnb_10s': 1, 'blues_10s': 2, 'hiphop_10s': 3, 'rock_10s': 4,
          'pop_20s': 0, 'rnb_20s': 1, 'blues_20s': 2, 'hiphop_20s': 3, 'rock_20s': 4
          }

data_5s = []
data_10s = []
data_20s = []


for name, path in folders_5s.items():
  #count_5s = 3000
  for filename in os.listdir(path):
    # if(count_5s == 0):
    #   break

    songData = []
    songname = f'{path}/{filename}'
    y, sr = librosa.load(songname, mono=True)

    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    songData.append(tempo)
    songData.append(mean(beats))

    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    songData.append(mean(chroma_stft))

    rmse = librosa.feature.rmse(y=y)
    songData.append(mean(rmse))

    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    songData.append(mean(spec_cent))

    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    songData.append(mean(spec_bw))

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    songData.append(mean(rolloff))

    zcr = librosa.feature.zero_crossing_rate(y)
    songData.append(mean(zcr))

    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    for i in mfcc:
        songData.append(mean(i))
    
    songData.append(label[name])

    data_5s.append(songData)

    #count_5s -= 1


for name, path in folders_10s.items():
  #count_10s = 1500
  for filename in os.listdir(path):
    # if(count_10s == 0):
    #   break

    songData = []
    songname = f'{path}/{filename}'
    y, sr = librosa.load(songname, mono=True)

    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    songData.append(tempo)
    songData.append(mean(beats))

    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    songData.append(mean(chroma_stft))

    rmse = librosa.feature.rmse(y=y)
    songData.append(mean(rmse))

    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    songData.append(mean(spec_cent))

    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    songData.append(mean(spec_bw))

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    songData.append(mean(rolloff))

    zcr = librosa.feature.zero_crossing_rate(y)
    songData.append(mean(zcr))

    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    for i in mfcc:
        songData.append(mean(i))
    
    songData.append(label[name])

    data_10s.append(songData)

    #count_10s -= 1


for name, path in folders_20s.items():
  #count_20s = 900
  for filename in os.listdir(path):
    # if(count_20s == 0):
    #   break

    songData = []
    songname = f'{path}/{filename}'
    y, sr = librosa.load(songname, mono=True)

    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    songData.append(tempo)
    songData.append(mean(beats))

    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    songData.append(mean(chroma_stft))

    rmse = librosa.feature.rmse(y=y)
    songData.append(mean(rmse))

    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    songData.append(mean(spec_cent))

    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    songData.append(mean(spec_bw))

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    songData.append(mean(rolloff))

    zcr = librosa.feature.zero_crossing_rate(y)
    songData.append(mean(zcr))

    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    for i in mfcc:
        songData.append(mean(i))
    
    songData.append(label[name])

    data_20s.append(songData)

    #count_20s -= 1




data_5s = pd.DataFrame(data_5s)
data_5s.to_csv('/content/drive/My Drive/ML_Project/data_5s_test_all_genres.csv') 

data_10s = pd.DataFrame(data_10s)
data_10s.to_csv('/content/drive/My Drive/ML_Project/data_10s_test_all_genres.csv') 

data_20s = pd.DataFrame(data_20s)
data_20s.to_csv('/content/drive/My Drive/ML_Project/data_20s_test_all_genres.csv')

data_10s

