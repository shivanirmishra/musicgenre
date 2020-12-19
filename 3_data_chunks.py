
from google.colab import drive
drive.mount('/content/drive')

import librosa
import librosa.display

data, sr = librosa.load('/content/drive/My Drive/Derek_Clegg_-_11_-_Heat.mp3')  
 
# To plot the pressure-time plot 
librosa.display.waveplot(data)

import matplotlib.pyplot as plt
import librosa
import librosa.display

data, sr = librosa.load('/content/drive/My Drive/Derek_Clegg_-_11_-_Heat.mp3', mono=False, duration=10)
plt.figure()
plt.subplot(3, 1, 2)
librosa.display.waveplot(data, sr=sr)
plt.title('Stereo')

plt.show()

data, sr = librosa.load('/content/drive/My Drive/Derek_Clegg_-_11_-_Heat.mp3') 
myaudio = AudioSegment.from_file('/content/drive/My Drive/Derek_Clegg_-_11_-_Heat.mp3' , "mp3") 
chunk_length_ms = 10000 # pydub calculates in millisec
chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec

#Export all of the individual chunks as wav files

for i, chunk in enumerate(chunks):
  chunk_name = "/content/drive/My Drive/song_pop{0}.mp3".format(i)
  chunk.export(chunk_name, format="mp3")

y = data
tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
rmse = librosa.feature.rmse(y=y)
spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
zcr = librosa.feature.zero_crossing_rate(y)
mfcc = librosa.feature.mfcc(y=y, sr=sr)
print(mfcc)

import librosa
import os
import pandas as pd
from numpy import median

label = {'pop': 0, 'rnb': 1}
data = []

for filename in os.listdir('/content/drive/My Drive/ML_Project/POP'):
    songData = []
    songname = f'/content/drive/My Drive/ML_Project/POP/{filename}'
    y, sr = librosa.load(songname, mono=True, duration=30)

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
        songData.append(np.mean(i))
    
    songData.append(label['pop'])

    data.append(songData)
    
    #data.append([tempo,mean(beats),mean(chroma_stft),mean(rmse),mean(spec_cent),mean(spec_bw),mean(rolloff),mean(zcr),mean(mfcc),'pop'])
    #data.append([tempo,median(beats),median(chroma_stft),median(rmse),median(spec_cent),median(spec_bw),median(rolloff),median(zcr),median(mfcc),'POP'])

for filename in os.listdir('/content/drive/My Drive/ML_Project/RnB'):
    songData = []
    songname = f'/content/drive/My Drive/ML_Project/RnB/{filename}'
    y, sr = librosa.load(songname, mono=True, duration=30)

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
        songData.append(np.mean(i))
    
    songData.append(label['rnb'])

    data.append(songData)

    #data.append([tempo,mean(beats),mean(chroma_stft),mean(rmse),mean(spec_cent),mean(spec_bw),mean(rolloff),mean(zcr),mean(mfcc),'RnB'])
    #data.append([tempo,median(beats),median(chroma_stft),median(rmse),median(spec_cent),median(spec_bw),median(rolloff),median(zcr),median(mfcc),'RnB'])    
data = pd.DataFrame(data)

print(np.mean(beats))

import numpy as np

data

data = data.sample(frac=1).reset_index(drop=True)

data

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import preprocessing

model = LogisticRegression(solver='lbfgs', max_iter=40000)
x = data.iloc[:,0:data.shape[1]-1]
y = data.iloc[:,data.shape[1]-1]

le = preprocessing.LabelEncoder()
y = le.fit_transform(y)

model.fit(x,y)
#y_pred_train = model.predict(x)
acc = model.score(x,y)

print(acc)

from pydub import AudioSegment
from pydub.utils import make_chunks
import os

j = 0

for filename in os.listdir('/content/drive/My Drive/ML_Project/POP'):
    songname = f'/content/drive/My Drive/ML_Project/POP/{filename}'
    myaudio = AudioSegment.from_file(songname , "mp3") 
    chunk_length_ms = 15000 # pydub calculates in millisec
    chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec

#Export all of the individual chunks as wav files

    for i, chunk in enumerate(chunks):
      chunk_name = "/content/drive/My Drive/ML_Project/New_Data/pop/song_pop{0}.mp3".format(i + j)
      chunk.export(chunk_name, format="mp3")
      j +=1

pip install pydub

j = 1

for filename in os.listdir('/content/drive/My Drive/ML_Project/RnB'):
    songname = f'/content/drive/My Drive/ML_Project/RnB/{filename}'
    myaudio = AudioSegment.from_file(songname , "mp3") 
    chunk_length_ms = 15000 # pydub calculates in millisec
    chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec

#Export all of the individual chunks as wav files

    for i, chunk in enumerate(chunks):
      chunk_name = "/content/drive/My Drive/ML_Project/New_Data/rnb/song_rnb{0}.mp3".format(i + j)
      chunk.export(chunk_name, format="mp3")
      j +=1

import librosa
import os
import pandas as pd
from numpy import mean

data = []
for filename in os.listdir('/content/drive/My Drive/ML_Project/New_Data/pop'):
    songname = f'/content/drive/My Drive/ML_Project/New_Data/pop/{filename}'
    y, sr = librosa.load(songname, mono=True, duration=30)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rmse = librosa.feature.rmse(y=y)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    data.append([tempo,mean(beats),mean(chroma_stft),mean(rmse),mean(spec_cent),mean(spec_bw),mean(rolloff),mean(zcr),mean(mfcc),'pop'])
    #data.append([tempo,median(beats),median(chroma_stft),median(rmse),median(spec_cent),median(spec_bw),median(rolloff),median(zcr),median(mfcc),'POP'])

for filename in os.listdir('/content/drive/My Drive/ML_Project/New_Data/rnb'):
    songname = f'/content/drive/My Drive/ML_Project/New_Data/rnb/{filename}'
    y, sr = librosa.load(songname, mono=True, duration=30)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rmse = librosa.feature.rmse(y=y)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    data.append([tempo,mean(beats),mean(chroma_stft),mean(rmse),mean(spec_cent),mean(spec_bw),mean(rolloff),mean(zcr),mean(mfcc),'RnB'])
    #data.append([tempo,median(beats),median(chroma_stft),median(rmse),median(spec_cent),median(spec_bw),median(rolloff),median(zcr),median(mfcc),'RnB'])    
data = pd.DataFrame(data)

data.to_csv('/content/drive/My Drive/ML_Project/data.csv')

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import preprocessing

data = data.dropna() 
model = LogisticRegression(solver='lbfgs', max_iter=4000)
x = data.iloc[:,0:data.shape[1]-1]
y = data.iloc[:,data.shape[1]-1]

le = preprocessing.LabelEncoder()
y = le.fit_transform(y)

model.fit(x,y)
#y_pred_train = model.predict(x)
acc = model.score(x,y)

acc

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import svm, datasets
import matplotlib.pyplot as plt
import numpy as np
x = data.iloc[:,0:data.shape[1]-1]
y = data.iloc[:,data.shape[1]-1]
linear = svm.SVC(kernel='linear', C=1, decision_function_shape='ovo').fit(x, y)
linear_pred = linear.predict(x)
accuracy_lin = linear.score(x, y)
print(accuracy_lin)

import librosa
import os
import pandas as pd
from numpy import mean
import warnings;
warnings.filterwarnings('ignore');

label = {'pop': 0, 'rnb': 1}
data = []

for filename in os.listdir('/content/drive/My Drive/ML_Project/New_Data/pop'):
    songData = []
    songname = f'/content/drive/My Drive/ML_Project/New_Data/pop/{filename}'
    y, sr = librosa.load(songname, mono=True, duration=30)

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
    
    songData.append(label['pop'])

    data.append(songData)
    
    #data.append([tempo,mean(beats),mean(chroma_stft),mean(rmse),mean(spec_cent),mean(spec_bw),mean(rolloff),mean(zcr),mean(mfcc),'pop'])
    #data.append([tempo,median(beats),median(chroma_stft),median(rmse),median(spec_cent),median(spec_bw),median(rolloff),median(zcr),median(mfcc),'POP'])

for filename in os.listdir('/content/drive/My Drive/ML_Project/New_Data/rnb'):
    songData = []
    songname = f'/content/drive/My Drive/ML_Project/New_Data/rnb/{filename}'
    y, sr = librosa.load(songname, mono=True, duration=30)

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
    
    songData.append(label['rnb'])

    data.append(songData)

    #data.append([tempo,mean(beats),mean(chroma_stft),mean(rmse),mean(spec_cent),mean(spec_bw),mean(rolloff),mean(zcr),mean(mfcc),'RnB'])
    #data.append([tempo,median(beats),median(chroma_stft),median(rmse),median(spec_cent),median(spec_bw),median(rolloff),median(zcr),median(mfcc),'RnB'])    

data = pd.DataFrame(data)
data.to_csv('/content/drive/My Drive/ML_Project/pop_rnb_data.csv')

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import preprocessing

data = data.dropna() 
model = LogisticRegression(solver='lbfgs', max_iter=4000)
x = data.iloc[:,0:data.shape[1]-1]
y = data.iloc[:,data.shape[1]-1]

le = preprocessing.LabelEncoder()
y = le.fit_transform(y)

model.fit(x,y)
#y_pred_train = model.predict(x)
acc = model.score(x,y)
print(acc)

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import svm, datasets
import matplotlib.pyplot as plt
import numpy as np
import librosa.display

x = data.iloc[:,0:data.shape[1]-1]
y = data.iloc[:,data.shape[1]-1]
linear = svm.SVC(kernel='linear', C=1, decision_function_shape='ovo').fit(x, y)
linear_pred = linear.predict(x)
accuracy_lin = linear.score(x, y)
print(accuracy_lin)

import librosa.display

y, sr = librosa.load('/content/drive/My Drive/Derek_Clegg_-_11_-_Heat.mp3')


# And compute the spectrogram magnitude and phase
S_full, phase = librosa.magphase(librosa.stft(y))
idx = slice(*librosa.time_to_frames([30, 35], sr=sr))
plt.figure(figsize=(12, 4))
librosa.display.specshow(librosa.amplitude_to_db(S_full[:, idx], ref=np.max),
                         y_axis='log', x_axis='time', sr=sr)
plt.colorbar()
plt.tight_layout()

S_filter = librosa.decompose.nn_filter(S_full,
                                       aggregate=np.median,
                                       metric='cosine',
                                       width=int(librosa.time_to_frames(2, sr=sr)))

# The output of the filter shouldn't be greater than the input
# if we assume signals are additive.  Taking the pointwise minimium
# with the input spectrum forces this.
S_filter = np.minimum(S_full, S_filter)
margin_i, margin_v = 2, 10
power = 2

mask_i = librosa.util.softmask(S_filter,
                               margin_i * (S_full - S_filter),
                               power=power)

mask_v = librosa.util.softmask(S_full - S_filter,
                               margin_v * S_filter,
                               power=power)

# Once we have the masks, simply multiply them with the input spectrum
# to separate the components

S_foreground = mask_v * S_full
S_background = mask_i * S_full

plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
librosa.display.specshow(librosa.amplitude_to_db(S_full[:, idx], ref=np.max),
                         y_axis='log', sr=sr)
plt.title('Full spectrum')
plt.colorbar()

plt.subplot(3, 1, 2)
librosa.display.specshow(librosa.amplitude_to_db(S_background[:, idx], ref=np.max),
                         y_axis='log', sr=sr)
plt.title('Background')
plt.colorbar()
plt.subplot(3, 1, 3)
librosa.display.specshow(librosa.amplitude_to_db(S_foreground[:, idx], ref=np.max),
                         y_axis='log', x_axis='time', sr=sr)
plt.title('Foreground')
plt.colorbar()
plt.tight_layout()
plt.show()

import librosa.output
new_y = librosa.istft(S_background*phase)
librosa.output.write_wav("/content/drive/My Drive/Derek_Clegg_-_11_-_Heat1.mp3", new_y, sr)

from pydub import AudioSegment
from pydub.playback import play

# read in audio file and get the two mono tracks
sound_stereo = AudioSegment.from_file('/content/drive/My Drive/Tropic_Island_-_Clearmix_ID_1172.mp3', format="mp3")
sound_monoL = sound_stereo.split_to_mono()[0]
sound_monoR = sound_stereo.split_to_mono()[1]

# Invert phase of the Right audio file
sound_monoR_inv = sound_monoR.invert_phase()

# Merge two L and R_inv files, this cancels out the centers
sound_CentersOut = sound_monoL.overlay(sound_monoR_inv)

# Export merged audio file
fh = sound_CentersOut.export('/content/drive/My Drive/Tropic_Island_-_Clearmix_ID_1172a.mp3', format="mp3")

sound_monoL

sound_monoR

sound_CentersOut

if(sound_monoL == sound_monoR):
  print('yes')

else:
  print('no')

from pydub import AudioSegment
from pydub.playback import play
import os

j = 1

for filename in os.listdir('/content/drive/My Drive/ML_Project/POP'):
    songname = f'/content/drive/My Drive/ML_Project/POP/{filename}'

# read in audio file and get the two mono tracks
    sound_stereo = AudioSegment.from_file(songname, format="mp3")
    sound_monoL = sound_stereo.split_to_mono()[0]
    sound_monoR = sound_stereo.split_to_mono()[1]

# Invert phase of the Right audio file
    sound_monoR_inv = sound_monoR.invert_phase()

# Merge two L and R_inv files, this cancels out the centers
    sound_CentersOut = sound_monoL.overlay(sound_monoR_inv)

# Export merged audio file
    fh = sound_CentersOut.export("/content/drive/My Drive/ML_Project/POP/song_pop{0}.mp3".format(j), format="mp3")
    j +=1

j = 1

for filename in os.listdir('/content/drive/My Drive/ML_Project/RnB'):
    songname = f'/content/drive/My Drive/ML_Project/RnB/{filename}'

# read in audio file and get the two mono tracks
    sound_stereo = AudioSegment.from_file(songname, format="mp3")
    sound_monoL = sound_stereo.split_to_mono()[0]
    sound_monoR = sound_stereo.split_to_mono()[1]

# Invert phase of the Right audio file
    sound_monoR_inv = sound_monoR.invert_phase()

# Merge two L and R_inv files, this cancels out the centers
    sound_CentersOut = sound_monoL.overlay(sound_monoR_inv)

# Export merged audio file
    fh = sound_CentersOut.export("/content/drive/My Drive/ML_Project/RnB/song_rnb{0}.mp3".format(j), format="mp3")
    j +=1

sound_monoR

sound_CentersOut

"""##**-------- Main Code --------**"""

pip install pydub

j = 1

for filename in os.listdir('/content/drive/My Drive/MLProject/hiphop'):
    songname = f'/content/drive/My Drive/MLProject/hiphop/{filename}'

# read in audio file and get the two mono tracks
    sound_stereo = AudioSegment.from_file(songname, format="mp3")
    sound_monoL = sound_stereo.split_to_mono()[0]
    sound_monoR = sound_stereo.split_to_mono()[1]

# Invert phase of the Right audio file
    sound_monoR_inv = sound_monoR.invert_phase()

# Merge two L and R_inv files, this cancels out the centers
    sound_CentersOut = sound_monoL.overlay(sound_monoR_inv)

# Export merged audio file
    fh = sound_CentersOut.export("/content/drive/My Drive/MLProject/hiphop/song_hip{0}.mp3".format(j), format="mp3")
    j +=1

j = 1

for filename in os.listdir('/content/drive/My Drive/MLProject/rock'):
    songname = f'/content/drive/My Drive/MLProject/rock/{filename}'

# read in audio file and get the two mono tracks
    sound_stereo = AudioSegment.from_file(songname, format="mp3")
    sound_monoL = sound_stereo.split_to_mono()[0]
    sound_monoR = sound_stereo.split_to_mono()[1]

# Invert phase of the Right audio file
    sound_monoR_inv = sound_monoR.invert_phase()

# Merge two L and R_inv files, this cancels out the centers
    sound_CentersOut = sound_monoL.overlay(sound_monoR_inv)

# Export merged audio file
    fh = sound_CentersOut.export("/content/drive/My Drive/MLProject/rock/song_rock{0}.mp3".format(j), format="mp3")
    j +=1

j = 1

for filename in os.listdir('/content/drive/My Drive/ML_Project/BLUES'):
    songname = f'/content/drive/My Drive/ML_Project/BLUES/{filename}'

# read in audio file and get the two mono tracks
    sound_stereo = AudioSegment.from_file(songname, format="mp3")
    sound_monoL = sound_stereo.split_to_mono()[0]
    sound_monoR = sound_stereo.split_to_mono()[1]

# Invert phase of the Right audio file
    sound_monoR_inv = sound_monoR.invert_phase()

# Merge two L and R_inv files, this cancels out the centers
    sound_CentersOut = sound_monoL.overlay(sound_monoR_inv)

# Export merged audio file
    fh = sound_CentersOut.export("/content/drive/My Drive/ML_Project/BLUES/song_blues{0}.mp3".format(j), format="mp3")
    j +=1

from pydub import AudioSegment
from pydub.utils import make_chunks
import os

j = 0

for filename in os.listdir('/content/drive/My Drive/ML_Project/POP'):
    songname = f'/content/drive/My Drive/ML_Project/POP/{filename}'
    myaudio = AudioSegment.from_file(songname , "mp3") 
    chunk_length_ms = 5000 # pydub calculates in millisec
    chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec

#Export all of the individual chunks as wav files

    for i, chunk in enumerate(chunks):
      chunk_name = "/content/drive/My Drive/ML_Project/New_Data/pop_5s/song_pop{0}.mp3".format(j)
      chunk.export(chunk_name, format="mp3")
      j +=1



for filename in os.listdir('/content/drive/My Drive/ML_Project/POP'):
    songname = f'/content/drive/My Drive/ML_Project/POP/{filename}'
    myaudio = AudioSegment.from_file(songname , "mp3") 
    chunk_length_ms = 10000 # pydub calculates in millisec
    chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec

#Export all of the individual chunks as wav files

    for i, chunk in enumerate(chunks):
      chunk_name = "/content/drive/My Drive/ML_Project/New_Data/pop_10s/song_pop{0}.mp3".format(j)
      chunk.export(chunk_name, format="mp3")
      j +=1



for filename in os.listdir('/content/drive/My Drive/ML_Project/POP'):
    songname = f'/content/drive/My Drive/ML_Project/POP/{filename}'
    myaudio = AudioSegment.from_file(songname , "mp3") 
    chunk_length_ms = 20000 # pydub calculates in millisec
    chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec

#Export all of the individual chunks as wav files

    for i, chunk in enumerate(chunks):
      chunk_name = "/content/drive/My Drive/ML_Project/New_Data/pop_20s/song_pop{0}.mp3".format(j)
      chunk.export(chunk_name, format="mp3")
      j +=1

for filename in os.listdir('/content/drive/My Drive/ML_Project/RnB'):
    songname = f'/content/drive/My Drive/ML_Project/RnB/{filename}'
    myaudio = AudioSegment.from_file(songname , "mp3") 
    chunk_length_ms = 5000 # pydub calculates in millisec
    chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec

#Export all of the individual chunks as wav files

    for i, chunk in enumerate(chunks):
      chunk_name = "/content/drive/My Drive/ML_Project/New_Data/rnb_5s/song_rnb{0}.mp3".format(j)
      chunk.export(chunk_name, format="mp3")
      j +=1



for filename in os.listdir('/content/drive/My Drive/ML_Project/RnB'):
    songname = f'/content/drive/My Drive/ML_Project/RnB/{filename}'
    myaudio = AudioSegment.from_file(songname , "mp3") 
    chunk_length_ms = 10000 # pydub calculates in millisec
    chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec

#Export all of the individual chunks as wav files

    for i, chunk in enumerate(chunks):
      chunk_name = "/content/drive/My Drive/ML_Project/New_Data/rnb_10s/song_rnb{0}.mp3".format(j)
      chunk.export(chunk_name, format="mp3")
      j +=1



for filename in os.listdir('/content/drive/My Drive/ML_Project/RnB'):
    songname = f'/content/drive/My Drive/ML_Project/RnB/{filename}'
    myaudio = AudioSegment.from_file(songname , "mp3") 
    chunk_length_ms = 20000 # pydub calculates in millisec
    chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec

#Export all of the individual chunks as wav files

    for i, chunk in enumerate(chunks):
      chunk_name = "/content/drive/My Drive/ML_Project/New_Data/rnb_20s/song_rnb{0}.mp3".format(j)
      chunk.export(chunk_name, format="mp3")
      j +=1

for filename in os.listdir('/content/drive/My Drive/ML_Project/BLUES'):
    songname = f'/content/drive/My Drive/ML_Project/BLUES/{filename}'
    myaudio = AudioSegment.from_file(songname , "mp3") 
    chunk_length_ms = 5000 # pydub calculates in millisec
    chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec

#Export all of the individual chunks as wav files

    for i, chunk in enumerate(chunks):
      chunk_name = "/content/drive/My Drive/ML_Project/New_Data/blues_5s/song_blues{0}.mp3".format(j)
      chunk.export(chunk_name, format="mp3")
      j +=1



for filename in os.listdir('/content/drive/My Drive/ML_Project/BLUES'):
    songname = f'/content/drive/My Drive/ML_Project/BLUES/{filename}'
    myaudio = AudioSegment.from_file(songname , "mp3") 
    chunk_length_ms = 10000 # pydub calculates in millisec
    chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec

#Export all of the individual chunks as wav files

    for i, chunk in enumerate(chunks):
      chunk_name = "/content/drive/My Drive/ML_Project/New_Data/blues_10s/song_blues{0}.mp3".format(j)
      chunk.export(chunk_name, format="mp3")
      j +=1



for filename in os.listdir('/content/drive/My Drive/ML_Project/BLUES'):
    songname = f'/content/drive/My Drive/ML_Project/BLUES/{filename}'
    myaudio = AudioSegment.from_file(songname , "mp3") 
    chunk_length_ms = 20000 # pydub calculates in millisec
    chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec

#Export all of the individual chunks as wav files

    for i, chunk in enumerate(chunks):
      chunk_name = "/content/drive/My Drive/ML_Project/New_Data/blues_20s/song_blues{0}.mp3".format(j)
      chunk.export(chunk_name, format="mp3")
      j +=1

from pydub import AudioSegment
from pydub.utils import make_chunks
import os

j = 0
for filename in os.listdir('/content/drive/My Drive/MLProject/hiphop'):
    songname = f'/content/drive/My Drive/MLProject/hiphop/{filename}'
    myaudio = AudioSegment.from_file(songname , "mp3") 
    chunk_length_ms = 5000 # pydub calculates in millisec
    chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec

#Export all of the individual chunks as wav files

    for i, chunk in enumerate(chunks):
      chunk_name = "/content/drive/My Drive/ML_Project/New_Data/hiphop_5s/song_hiphop{0}.mp3".format(j)
      chunk.export(chunk_name, format="mp3")
      j +=1



for filename in os.listdir('/content/drive/My Drive/MLProject/hiphop'):
    songname = f'/content/drive/My Drive/MLProject/hiphop/{filename}'
    myaudio = AudioSegment.from_file(songname , "mp3") 
    chunk_length_ms = 10000 # pydub calculates in millisec
    chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec

#Export all of the individual chunks as wav files

    for i, chunk in enumerate(chunks):
      chunk_name = "/content/drive/My Drive/ML_Project/New_Data/hiphop_10s/song_hiphop{0}.mp3".format(j)
      chunk.export(chunk_name, format="mp3")
      j +=1



for filename in os.listdir('/content/drive/My Drive/MLProject/hiphop'):
    songname = f'/content/drive/My Drive/MLProject/hiphop/{filename}'
    myaudio = AudioSegment.from_file(songname , "mp3") 
    chunk_length_ms = 20000 # pydub calculates in millisec
    chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec

#Export all of the individual chunks as wav files

    for i, chunk in enumerate(chunks):
      chunk_name = "/content/drive/My Drive/ML_Project/New_Data/hiphop_20s/song_hiphop{0}.mp3".format(j)
      chunk.export(chunk_name, format="mp3")
      j +=1

for filename in os.listdir('/content/drive/My Drive/MLProject/rock'):
    songname = f'/content/drive/My Drive/MLProject/rock/{filename}'
    myaudio = AudioSegment.from_file(songname , "mp3") 
    chunk_length_ms = 5000 # pydub calculates in millisec
    chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec

#Export all of the individual chunks as wav files

    for i, chunk in enumerate(chunks):
      chunk_name = "/content/drive/My Drive/ML_Project/New_Data/rock_5s/song_rock{0}.mp3".format(j)
      chunk.export(chunk_name, format="mp3")
      j +=1



for filename in os.listdir('/content/drive/My Drive/MLProject/rock'):
    songname = f'/content/drive/My Drive/MLProject/rock/{filename}'
    myaudio = AudioSegment.from_file(songname , "mp3")
    chunk_length_ms = 10000 # pydub calculates in millisec
    chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec

#Export all of the individual chunks as wav files

    for i, chunk in enumerate(chunks):
      chunk_name = "/content/drive/My Drive/ML_Project/New_Data/rock_10s/song_rock{0}.mp3".format(j)
      chunk.export(chunk_name, format="mp3")
      j +=1



for filename in os.listdir('/content/drive/My Drive/MLProject/rock'):
    songname = f'/content/drive/My Drive/MLProject/rock/{filename}'
    myaudio = AudioSegment.from_file(songname , "mp3") 
    chunk_length_ms = 20000 # pydub calculates in millisec
    chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec

#Export all of the individual chunks as wav files

    for i, chunk in enumerate(chunks):
      chunk_name = "/content/drive/My Drive/ML_Project/New_Data/rock_20s/song_rock{0}.mp3".format(j)
      chunk.export(chunk_name, format="mp3")
      j +=1

pip install pydub

from tabulate import tabulate

folders = {
           'pop_5s':'/content/drive/My Drive/ML_Project/New_Data/pop_5s', 
           'pop_10s':'/content/drive/My Drive/ML_Project/New_Data/pop_10s',
           'pop_20s':'/content/drive/My Drive/ML_Project/New_Data/pop_20s',
           'rnb_5s':'/content/drive/My Drive/ML_Project/New_Data/rnb_5s', 
           'rnb_10s':'/content/drive/My Drive/ML_Project/New_Data/rnb_10s',
           'rnb_20s':'/content/drive/My Drive/ML_Project/New_Data/rnb_20s',
           'blues_5s':'/content/drive/My Drive/ML_Project/New_Data/blues_5s', 
           'blues_10s':'/content/drive/My Drive/ML_Project/New_Data/blues_10s',
           'blues_20s':'/content/drive/My Drive/ML_Project/New_Data/blues_20s',
           'hiphop_5s':'/content/drive/My Drive/ML_Project/New_Data/hiphop_5s', 
           'hiphop_10s':'/content/drive/My Drive/ML_Project/New_Data/hiphop_10s',
           'hiphop_20s':'/content/drive/My Drive/ML_Project/New_Data/hiphop_20s',
           'rock_5s':'/content/drive/My Drive/ML_Project/New_Data/rock_5s', 
           'rock_10s':'/content/drive/My Drive/ML_Project/New_Data/rock_10s',
           'rock_20s':'/content/drive/My Drive/ML_Project/New_Data/rock_20s'
           }
count_samples = {
                 'pop_5s':0,
                 'pop_10s':0,
                 'pop_20s':0,
                 'rnb_5s':0,
                 'rnb_10s':0,
                 'rnb_20s':0,
                 'blues_5s':0,
                 'blues_10s':0,
                 'blues_20s':0,
                 'hiphop_5s':0,
                 'hiphop_10s':0,
                 'hiphop_20s':0,
                 'rock_5s':0,
                 'rock_10s':0,
                 'rock_20s':0
                 }

for name,path in folders.items():           
  for filename in os.listdir(path):
    count_samples[name] +=1


headers = ['Genre with length','count']
print(tabulate(count_samples.items(), headers=headers, tablefmt = "fancy_grid"))

import librosa
import os
import pandas as pd
from numpy import mean
import warnings;
warnings.filterwarnings('ignore');

folders_5s = {
              'pop_5s':'/content/drive/My Drive/ML_Project/New_Data/pop_5s',
              'rnb_5s':'/content/drive/My Drive/ML_Project/New_Data/rnb_5s', 
              'blues_5s':'/content/drive/My Drive/ML_Project/New_Data/blues_5s',
              'hiphop_5s':'/content/drive/My Drive/ML_Project/New_Data/hiphop_5s',
              'rock_5s':'/content/drive/My Drive/ML_Project/New_Data/rock_5s'
            }

folders_10s = {
              'pop_10s':'/content/drive/My Drive/ML_Project/New_Data/pop_10s',
              'rnb_10s':'/content/drive/My Drive/ML_Project/New_Data/rnb_10s', 
              'blues_10s':'/content/drive/My Drive/ML_Project/New_Data/blues_10s',
              'hiphop_10s':'/content/drive/My Drive/ML_Project/New_Data/hiphop_10s',
              'rock_10s':'/content/drive/My Drive/ML_Project/New_Data/rock_10s'
            }

folders_20s = {
              'pop_20s':'/content/drive/My Drive/ML_Project/New_Data/pop_20s',
              'rnb_20s':'/content/drive/My Drive/ML_Project/New_Data/rnb_20s', 
              'blues_20s':'/content/drive/My Drive/ML_Project/New_Data/blues_20s',
              'hiphop_20s':'/content/drive/My Drive/ML_Project/New_Data/hiphop_20s',
              'rock_20s':'/content/drive/My Drive/ML_Project/New_Data/rock_20s'
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
  count_5s = 3000
  for filename in os.listdir(path):
    if(count_5s == 0):
      break

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

    count_5s -= 1


# for name, path in folders_10s.items():
#   count_10s = 1500
#   for filename in os.listdir(path):
#     if(count_10s == 0):
#       break

#     songData = []
#     songname = f'{path}/{filename}'
#     y, sr = librosa.load(songname, mono=True)

#     tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
#     songData.append(tempo)
#     songData.append(mean(beats))

#     chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
#     songData.append(mean(chroma_stft))

#     rmse = librosa.feature.rmse(y=y)
#     songData.append(mean(rmse))

#     spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
#     songData.append(mean(spec_cent))

#     spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
#     songData.append(mean(spec_bw))

#     rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
#     songData.append(mean(rolloff))

#     zcr = librosa.feature.zero_crossing_rate(y)
#     songData.append(mean(zcr))

#     mfcc = librosa.feature.mfcc(y=y, sr=sr)
#     for i in mfcc:
#         songData.append(mean(i))
    
#     songData.append(label[name])

#     data_10s.append(songData)

#     count_10s -= 1


# for name, path in folders_20s.items():
#   count_20s = 900
#   for filename in os.listdir(path):
#     if(count_20s == 0):
#       break

#     songData = []
#     songname = f'{path}/{filename}'
#     y, sr = librosa.load(songname, mono=True)

#     tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
#     songData.append(tempo)
#     songData.append(mean(beats))

#     chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
#     songData.append(mean(chroma_stft))

#     rmse = librosa.feature.rmse(y=y)
#     songData.append(mean(rmse))

#     spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
#     songData.append(mean(spec_cent))

#     spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
#     songData.append(mean(spec_bw))

#     rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
#     songData.append(mean(rolloff))

#     zcr = librosa.feature.zero_crossing_rate(y)
#     songData.append(mean(zcr))

#     mfcc = librosa.feature.mfcc(y=y, sr=sr)
#     for i in mfcc:
#         songData.append(mean(i))
    
#     songData.append(label[name])

#     data_20s.append(songData)

#     count_20s -= 1




data_5s = pd.DataFrame(data_5s)
data_5s.to_csv('/content/drive/My Drive/ML_Project/data_5s_all_genres.csv') 

# data_10s = pd.DataFrame(data_10s)
# data_10s.to_csv('/content/drive/My Drive/ML_Project/data_10s_all_genres.csv') 

# data_20s = pd.DataFrame(data_20s)
# data_20s.to_csv('/content/drive/My Drive/ML_Project/data_20s_all_genres.csv')

"""#Preparing test data chunks"""

from pydub import AudioSegment
from pydub.utils import make_chunks
import os

j = 0

for filename in os.listdir('/content/drive/My Drive/ML_Project/rnb_test'):
    songname = f'/content/drive/My Drive/ML_Project/rnb_test/{filename}'
    myaudio = AudioSegment.from_file(songname , "mp3") 
    chunk_length_ms = 5000 # pydub calculates in millisec
    chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec

#Export all of the individual chunks as wav files

    for i, chunk in enumerate(chunks):
      chunk_name = "/content/drive/My Drive/ML_Project/New_Data/rnb_test_5s/song_rn{0}.mp3".format(j)
      chunk.export(chunk_name, format="mp3")
      j +=1



for filename in os.listdir('/content/drive/My Drive/ML_Project/rnb_test'):
    songname = f'/content/drive/My Drive/ML_Project/rnb_test/{filename}'
    myaudio = AudioSegment.from_file(songname , "mp3") 
    chunk_length_ms = 10000 # pydub calculates in millisec
    chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec

#Export all of the individual chunks as wav files

    for i, chunk in enumerate(chunks):
      chunk_name = "/content/drive/My Drive/ML_Project/New_Data/rnb_test_10s/song_rn{0}.mp3".format(j)
      chunk.export(chunk_name, format="mp3")
      j +=1




for filename in os.listdir('/content/drive/My Drive/ML_Project/rnb_test'):
    songname = f'/content/drive/My Drive/ML_Project/rnb_test/{filename}'
    myaudio = AudioSegment.from_file(songname , "mp3") 
    chunk_length_ms = 15000 # pydub calculates in millisec
    chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec

#Export all of the individual chunks as wav files

    for i, chunk in enumerate(chunks):
      chunk_name = "/content/drive/My Drive/ML_Project/New_Data/rnb_test_15s/song_rn{0}.mp3".format(j)
      chunk.export(chunk_name, format="mp3")
      j +=1


for filename in os.listdir('/content/drive/My Drive/ML_Project/rnb_test'):
    songname = f'/content/drive/My Drive/ML_Project/rnb_test/{filename}'
    myaudio = AudioSegment.from_file(songname , "mp3") 
    chunk_length_ms = 20000 # pydub calculates in millisec
    chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec

#Export all of the individual chunks as wav files

    for i, chunk in enumerate(chunks):
      chunk_name = "/content/drive/My Drive/ML_Project/New_Data/rnb_test_20s/song_rn{0}.mp3".format(j)
      chunk.export(chunk_name, format="mp3")
      j +=1

pip install pydub

