# Автор: Стрельников Вячеслав Евгеньевич, группа 6305.
import os, glob
import numpy as np
import pandas as pd
import librosa as l

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Расчет датафрейма параметров MFCC
def compute_mfcc(path):
    MFCClist = []
    files = os.listdir(path)
    for filename in glob.glob(os.path.join(path, '*.wav')):
        x, sr = l.load(filename)
        mfccs = l.feature.mfcc(x, sr=sr)
        data = [np.mean(feature) for feature in mfccs]
        MFCClist.append(data)
    df = pd.DataFrame(MFCClist)
    df.columns = ['MFCC_0', 'MFCC_1', 'MFCC_2', 'MFCC_3', 'MFCC_4', 'MFCC_5', 'MFCC_6', 'MFCC_7', 'MFCC_8', 'MFCC_9', 'MFCC_10', 'MFCC_11', 'MFCC_12', 'MFCC_13', 'MFCC_14', 'MFCC_15', 'MFCC_16', 'MFCC_17', 'MFCC_18', 'MFCC_19']
    return df

# Расчет датафрейма параметров MFCD
def compute_mfcc_delta(path):
    MFCClist = []
    files = os.listdir(path)
    for filename in glob.glob(os.path.join(path, '*.wav')):
        x, sr = l.load(filename)
        mfccs = l.feature.mfcc(x, sr=sr)
        mfccs_delta = l.feature.delta(mfccs)
        data = [np.mean(feature) for feature in mfccs_delta]
        MFCClist.append(data)
    df = pd.DataFrame(MFCClist)
    df.columns = ['MFCD_0', 'MFCD_1', 'MFCD_2', 'MFCD_3', 'MFCD_4', 'MFCD_5', 'MFCD_6', 'MFCD_7', 'MFCD_8', 'MFCD_9', 'MFCD_10', 'MFCD_11', 'MFCD_12', 'MFCD_13', 'MFCD_14', 'MFCD_15', 'MFCD_16', 'MFCD_17', 'MFCD_18', 'MFCD_19']
    return df

# Выполнение метода голавных компонент над датафреймов
def pca_dataframe(signalDataFrame, n_components):
    pca = PCA(n_components)
    XPCAreduced = pca.fit_transform(signalDataFrame)
    XPCAreduced = pd.DataFrame(data=XPCAreduced)
    return XPCAreduced