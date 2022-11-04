#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from pathlib import Path
import IPython.display as ipd
# % pylab inline
import os


import librosa
import glob 
import librosa.display
import random

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics 

from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout 
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

from keras.callbacks import EarlyStopping

from keras import regularizers

from sklearn.preprocessing import LabelEncoder

from datetime import datetime


# In[2]:


filelist = os.listdir('C:\\Users\\ashwi\\Desktop\\Test_speech') 
print(filelist)
df_p1 = pd.DataFrame(filelist)


# In[3]:


df_p1['label']='1'
# Renaming the column name to file
df_p1 = df_p1.rename(columns={0:'file'})
df_p1.head()
print(df_p1)
df_p1[df_p1['file']=='.DS_Store']

filelist = os.listdir('C:\\Users\\ashwi\\Desktop\\Test_speech1') 
print(filelist)
df_p2 = pd.DataFrame(filelist)

# Adding the 1 label to the dataframe representing male
df_p2['label']='2'
# Renaming the column name to file
df_p2 = df_p2.rename(columns={0:'file'})
df_p2.head()
#print(df_p2)
df_p2[df_p2['file']=='.DS_Store']
df_p2 = df_p2.reset_index(drop=True)


df = pd.concat([df_p2, df_p1], ignore_index=True)
df.head()
df = df.sample(frac=1).reset_index(drop=True)
#print(df)


df_train = df[0:6]
df_train['label'].value_counts(normalize=True)
#print(df_train)

df_validation = df[6:8]
df_validation['label'].value_counts(normalize=True)
#print(df_validation)

df_test = df_validation[8:]
df_test['label'].value_counts(normalize=True)


# In[4]:


def extract_features(files):
    
    # Sets the name to be the path to where the file is in my computer
    file_name = os.path.join(os.path.abspath('C:\\Users\\ashwi\\Desktop\\voice')+'//'+str(files.file))

    # Loads the audio file as a floating point time series and assigns the default sample rate
    # Sample rate is set to 22050 by default
    X, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 

    # Generate Mel-frequency cepstral coefficients (MFCCs) from a time series 
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)

    # Generates a Short-time Fourier transform (STFT) to use in the chroma_stft
    stft = np.abs(librosa.stft(X))

    # Computes a chromagram from a waveform or power spectrogram.
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)

    # Computes a mel-scaled spectrogram.
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)

    # Computes spectral contrast
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)

    # Computes the tonal centroid features (tonnetz)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
    sr=sample_rate).T,axis=0)
        
    
    # We add also the classes of each file as a label at the end
   # label = files.label

    return mfccs, chroma, mel, contrast, tonnetz


# In[5]:


features_label = df.apply(extract_features, axis=1)
print(features_label)


# In[6]:


features = []
for i in range(0, len(features_label)):
    features.append(np.concatenate((features_label[i][0], features_label[i][1], 
                features_label[i][2], features_label[i][3],
                features_label[i][4]), axis=0))
    
print(len(features))


# In[7]:


#print(features)


# In[8]:


speaker = []
for i in range(0, len(df)):
    speaker.append(df['file'][i].split('-')[0])


# In[9]:


df['speaker'] = speaker


# In[10]:


df.head()


# In[11]:


df['speaker'].nunique()


# In[12]:


labels = speaker


# In[13]:


np.unique(labels, return_counts=True)


# In[14]:


X = np.array(features)
y = np.array(labels)
# Hot encoding y
lb = LabelEncoder()
y = to_categorical(lb.fit_transform(y))
X.shape


# In[15]:


y.shape
# Choosing the first 9188 (70%) files to be our train data
# Choosing the next  2625 (20%) files to be our validation data
# Choosing the next  1312 (10%) files to be our test never before seen data
# This is analogous to a train test split but we add a validation split and we are making
# we do not shuffle anything since we are dealing with several time series, we already 
# checked before that we have balanced classes (analogous to stratify)

X_train = X[:6]
y_train = y[:6]

X_val = X[6:8]
y_val = y[6:8]

X_test = X[8:]
y_test = y[8:]
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_val = ss.transform(X_val)
X_test = ss.transform(X_test)


# In[24]:


model = Sequential()

model.add(Dense(2, input_shape=(193,), activation = 'relu'))
model.add(Dropout(0.1))

model.add(Dense(2, activation = 'relu'))
model.add(Dropout(0.25))  

model.add(Dense(2, activation = 'relu'))
model.add(Dropout(0.5))    

model.add(Dense(2, activation = 'sigmoid'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='auto')


# In[25]:


history = model.fit(X_train, y_train, batch_size=12, epochs=100, 
                    validation_data=(X_val, y_val))
                    


# In[ ]:




