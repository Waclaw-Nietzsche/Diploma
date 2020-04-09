from __future__ import print_function, division
from wave import open as open_wave
from IPython.display import Audio
#%matplotlib inline

import array
import copy
import math
import cython
import random
import scipy as sp
import scipy.stats
from scipy import signal
import scipy.fftpack
import struct
import subprocess
import thinkplot
import warnings
import os, glob
import seaborn as sn
warnings.simplefilter("ignore", DeprecationWarning)

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10,5)
import seaborn  as sns
import pandas as pd
from pandas import read_csv
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
import soundmfccparams as smfccp

# Считать WAV-file
def read_wave(filename='sound.wav'):
        fp = open_wave(filename, 'r')
        # Число аудиоканалов
        nchannels = fp.getnchannels()
        # Число аудиокадров
        nframes = fp.getnframes()
        # Ширина выборки (в байтах)
        sampwidth = fp.getsampwidth()
        # Частота дискретизации
        framerate = fp.getframerate()
        # Читает и возвращает не более n кадров аудио как объект байтов.
        z_str = fp.readframes(nframes)

        fp.close()

        dtype_map = {1:np.int8, 2:np.int16, 3:'special', 4:np.int32}
        if sampwidth not in dtype_map:
            raise ValueError('sampwidth %d unknown' % sampwidth)

        if sampwidth == 3:
            xs = np.fromstring(z_str, dtype=np.int8).astype(np.int32)
            ys = (xs[2::3] * 256 + xs[1::3]) * 256 + xs[0::3]
        else:
            ys = np.fromstring(z_str, dtype=dtype_map[sampwidth])

        # Если это стерео-сигнал, то нам будет достаточно одного сигнала
        if nchannels == 2:
            ys = ys[::2]

        #ts = np.arange(len(ys)) / framerate
        wave = Wave(ys, framerate=framerate)
        #wave.normalize()
        return wave

class Spectrogram:
    """Represents the spectrum of a signal."""

    def __init__(self, spec_map, seg_length):
        """Initialize the spectrogram.

        spec_map: map from float time to Spectrum
        seg_length: number of samples in each segment
        """
        self.spec_map = spec_map
        self.seg_length = seg_length

    def any_spectrum(self):
        """Returns an arbitrary spectrum from the spectrogram."""
        index = next(iter(self.spec_map))
        return self.spec_map[index]

    @property
    def time_res(self):
        """Time resolution in seconds."""
        spectrum = self.any_spectrum()
        return float(self.seg_length) / spectrum.framerate

    @property
    def freq_res(self):
        """Frequency resolution in Hz."""
        return self.any_spectrum().freq_res

    def times(self):
        """Sorted sequence of times.

        returns: sequence of float times in seconds
        """
        ts = sorted(iter(self.spec_map))
        return ts

    def frequencies(self):
        """Sequence of frequencies.

        returns: sequence of float freqencies in Hz.
        """
        fs = self.any_spectrum().fs
        return fs

    def plot(self, high=None, **options):
        """Make a pseudocolor plot.

        high: highest frequency component to plot
        """
        fs = self.frequencies()
        i = None if high is None else find_index(high, fs)
        fs = fs[:i]
        ts = self.times()

        # make the array
        size = len(fs), len(ts)
        array = np.zeros(size, dtype=np.float)

        # copy amplitude from each spectrum into a column of the array
        for j, t in enumerate(ts):
            spectrum = self.spec_map[t]
            array[:, j] = spectrum.amps[:i]

        thinkplot.pcolor(ts, fs, array, **options)

    def make_wave(self):
        """Inverts the spectrogram and returns a Wave.

        returns: Wave
        """
        res = []
        for t, spectrum in sorted(self.spec_map.items()):
            wave = spectrum.make_wave()
            n = len(wave)
            
            window = 1 / np.hamming(n)
            wave.window(window)

            i = wave.find_index(t)
            start = i - n // 2
            end = start + n
            res.append((start, end, wave))

        starts, ends, waves = zip(*res)
        low = min(starts)
        high = max(ends)

        ys = np.zeros(high-low, np.float)
        for start, end, wave in res:
            ys[start:end] = wave.ys

        # ts = np.arange(len(ys)) / self.framerate
        return Wave(ys, framerate=wave.framerate)

# Найти индекс элемента x  в массиве xs
def find_index(x, xs):
    n = len(xs)
    start = xs[0]
    end = xs[-1]
    i = round((n-1) * (x - start) / (end - start))
    return int(i)

class Wave:
    def __init__(self, ys, ts=None, framerate=None):

        #ys: массив значений wave сигнала
        #ts: массив временных интервалов
        #framerate: элементов выборки в секунду    
        self.ys = np.asanyarray(ys) 
        
        # Стандартные частоты дискретизации для звуковых плат - 11,025, 22,05 и 44,1 КГц. а разрядности - 8,12 и 16. (возьмём первую, если не указано)
        self.framerate = framerate if framerate is not None else 11025
        
        # Общая длительность по времени (если не указано, вычисляем)
        if ts is None:
            self.ts = np.arange(len(ys)) / self.framerate
        else:
            self.ts = np.asanyarray(ts)
            
    # Конструктор копии (глубокая) - создает новый составной объект, и затем рекурсивно вставляет в него копии объектов, находящихся в оригинале
    def copy(self):
        return copy.deepcopy(self)
    
    # Длина массива сигнала
    def __len__(self):
        return len(self.ys)

    def __add__(self, other):
        """
        Вычисляет среднее значение.
        """
        if other == 0:
            return self

        assert self.framerate == other.framerate

        # make an array of times that covers both waves
        start = min(self.start, other.start)
        end = max(self.end, other.end)
        n = int(round((end - start) * self.framerate)) + 1
        ys = np.zeros(n)
        ts = start + np.arange(n) / self.framerate

        def add_ys(wave):
            i = find_index(wave.start, ts)

            # make sure the arrays line up reasonably well
            diff = ts[i] - wave.start
            dt = 1 / wave.framerate
            if (diff / dt) > 0.1:
                warnings.warn("Can't add these waveforms; their "
                              "time arrays don't line up.")

            j = i + len(wave)
            ys[i:j] += wave.ys
            
        def add_y_mid(wave):
            i = find_index(wave.start, ts)

            # make sure the arrays line up reasonably well
            diff = ts[i] - wave.start
            dt = 1 / wave.framerate
            if (diff / dt) > 0.1:
                warnings.warn("Can't add these waveforms; their "
                              "time arrays don't line up.")

            j = i + len(wave)
            ys[i:j] += wave.ys/2

        add_y_mid(self)
        add_y_mid(other)

        return Wave(ys, ts, self.framerate)
    
## БАЗОВЫЕ ВЫЧИСЛЯЕМЫЕ СВОЙСТВА
    # Отсчёт времени от нуля
    @property
    def start(self):
        return self.ts[0]
    
    # До достижения конца времени звучания
    @property
    def end(self):
        return self.ts[-1]
    
    # Длительность (в секундах) (float)
    @property
    def duration(self):
        return len(self.ys) / self.framerate
    
    # Выравнивание сигнала по амплитуде (максимальная амплитуда – фиксированная и равна единице)
    def normalize(self, amp=1.0):
        self.ys = normalize(self.ys, amp=amp)
    
    # Найти индекс по заданной величине из массива
    def find_index(self, t):
        n = len(self)
        start = self.start
        end = self.end
        i = round((n-1) * (t - start) / (end - start))
        return int(i)
    
    # Делаем срез сегмента сигнала от i до j (индексы среза)
    def slicen(self, i, j):
        ys = self.ys[i:j].copy()
        ts = self.ts[i:j].copy()
        return Wave(ys, ts, self.framerate)
    
    # Вычленяем сегмент сигнала по времени (ОТ - ДО)
    def segment(self, start=None, duration=None):
        if start is None:
            start = self.ts[0]
            i = 0
        else:
            i = self.find_index(start)

        j = None if duration is None else self.find_index(start + duration)
        return self.slicen(i, j)
    
    # Построение спектра с помощью быстрого преобразования Фурье
    def make_spectrum(self):
        n = len(self.ys) # Длина
        d = 1 / self.framerate # Интервал выборки
        # Вычислить одномерное дискретное преобразование Фурье
        hs = np.fft.rfft(self.ys)
        # Частоты дискретизации дискретного преобразования Фурье
        fs = np.fft.rfftfreq(n, d)
        return Spectrum(hs, fs, self.framerate)
    
    def make_split_spectrum(self):
        splt = np.array_split(self.ys, 4)
        i = 0
        d = (1 / self.framerate) / 4 # Интервал выборки
        spec_list = []
        while (i != 4):
            n = len(splt[i])
            # Вычислить одномерное дискретное преобразование Фурье
            hs = np.fft.rfft(splt[i])
            # Частоты дискретизации дискретного преобразования Фурье
            fs = np.fft.rfftfreq(n, d)
            spec_list.append(Spectrum(hs, fs, self.framerate))
            i = i + 1
        return spec_list
    
    # Вычисление центроида spectrum centroid
    def calculate_centroid(self):
        magnitudes = np.abs(np.fft.rfft(self.ys))
        length = len(self.ys)
        freqs = np.abs(np.fft.fftfreq(length, 1.0/self.framerate)[:length//2+1])
        return np.sum(magnitudes*freqs) / np.sum(magnitudes)
    
    # Вычисление spectrum spread
    def calculate_ss(self):
        n = len(self.ys) # Длина
        d = 1 / self.framerate # Интервал выборки
        # Вычислить одномерное дискретное преобразование Фурье
        hs = np.abs(np.fft.rfft(self.ys))
        # Частоты дискретизации дискретного преобразования Фурье
        fs = np.abs(np.fft.rfftfreq(n, d))
        return np.sum((fs - self.calculate_centroid())**2 *hs) / np.sum(hs)
    
        # Вычисление spectrum flatness
    def calculate_sf(self):
        length = len(self.ys)
        magnitudes = np.abs(np.fft.rfft(self.ys))
        return (np.exp((1 / length) * np.sum(np.log(magnitudes)))) / ((1 / length) * np.sum(magnitudes))
    
    def make_audio(self):
        audio = Audio(data=self.ys.real, rate=self.framerate)
        return audio
    
    def make_spectrogram(self, seg_length):
        #seg_length: количество элементов в каждом сегменте
        
        i, j = 0, seg_length
        step = int(seg_length // 2)

        # map from time to Spectrum
        spec_map = {}

        while j < len(self.ys):
            segment = self.slicen(i, j)

            # the nominal time for this segment is the midpoint
            t = (segment.start + segment.end) / 2
            spec_map[t] = segment.make_spectrum()

            i += step
            j += step

        return Spectrogram(spec_map, seg_length)
    
    def plot_spectrogram(self, wave, seg_length):
        spectrogram = wave.make_spectrogram(seg_length)
        print('Time resolution (s)', spectrogram.time_res)
        print('Frequency resolution (Hz)', spectrogram.freq_res)
        spectrogram.plot()
        thinkplot.show(xlabel='Time(s)', ylabel='Frequency (Hz)')
        return spectrogram    
    
    
    def plot(self, **options):
        thinkplot.plot(self.ts, self.ys, **options)
        
# Нормализация по максимальной амплитуде (в данном случае 1-ца)
def normalize(ys, amp=1.0):
    high, low = abs(max(ys)), abs(min(ys))
    return amp * ys / max(high, low)

class Spectrum:
    def __init__(self, hs, fs, framerate):
        self.hs = np.asanyarray(hs)
        self.fs = np.asanyarray(fs)
        self.framerate = framerate
    
    @property
    def max_freq(self):
        # Частота Найквиста (половина частоты Дискретизации)
        return self.framerate / 2

    @property
    def freq_res(self):
        return self.framerate / 2 / (len(self.fs) - 1)
    
    @property
    def real(self):
        """Returns the real part of the hs (read-only property)."""
        return np.real(self.hs)

    @property
    def imag(self):
        """Returns the imaginary part of the hs (read-only property)."""
        return np.imag(self.hs)
    
    @property
    def amps(self):
        # Последовательность амплитуд
        return np.absolute(self.hs)
    
    @property
    def freq(self):
        # Последовательность частот
        return self.fs
    
    @property
    def angles(self):
        """Returns a sequence of angles (read-only property)."""
        return np.angle(self.hs)

    @property
    def power(self):
        # Последовательность мощностей
        return self.amps ** 2
    
    # Строим график Амплитуды по отношению к частоте (high - фильтр, если спектр полный, ограничение игнорируется)
    def plot(self, high=None, **options):
        i = None if high is None else find_index(high, self.fs)
        thinkplot.plot(self.fs[:i], self.amps[:i], **options)
    
    # Строим график Мощности по отношению к частоте
    def plot_power(self, high=None, **options):
            i = None if high is None else find_index(high, self.fs)
            thinkplot.plot(self.fs[:i], self.power[:i], **options)
            
class SP (object):
          
    def create_wave(self, name='sound.wav'):    
        wave = read_wave(name)
        wave.plot(color='dodgerblue')
        thinkplot.config(xlabel='Time (s)')
        thinkplot.config(ylabel='Max Amplitude')
        return wave

    def create_seg(self, file, start, duration):
        segm = file.segment(start,duration)
        segm.plot(color='deepskyblue')
        thinkplot.config(xlabel='Time (s)')
        thinkplot.config(ylabel='Max Amplitude')
        return segm
    
    def create_spect(self, name):
        spect = name.make_spectrum()
        spect.plot(color='red')
        thinkplot.config(xlabel='Frequency (Hz)')
        thinkplot.config(ylabel='Amplitude')
        return spect

def rms_flat(a):
    """
    Return the root mean square of all the elements of *a*, flattened out.
    """
    return np.sqrt(np.mean(abs(a)**2))
     
def mean_confidence_interval(data, confidence=0.99):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    h = h*3 # Расширяем интервал до 3-х СКО
    return m-h, m, m+h

def deviation(data, confidence=0.99):
    a = 1.0*np.array(data)
    n = len(a)
    se = scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return h

def calculate_mean_amplitude_list(path="C:/Users/tester/Desktop/Strelnikov/noise1"):
    alist = []
    files = os.listdir(path)
    for filename in glob.glob(os.path.join(path, '*.wav')):
        wave = read_wave(filename)
        spect = wave.make_spectrum()
        node = spect.amps
        average = np.average(node)
        alist.append(average)
    return alist

#split
def split_calculate_mean_amplitude_list(path="C:/Users/tester/Desktop/Strelnikov/noise1"):
    alist = []
    files = os.listdir(path)
    for filename in glob.glob(os.path.join(path, '*.wav')):
        temp = []
        wave = read_wave(filename)
        spect = wave.make_split_spectrum()
        j = 0
        while (j != 4):
            node = spect[j].amps
            average = np.average(node)
            temp.append(average)
            j = j + 1
        alist.append(temp)
    return alist

def calculate_SCO_amplitude_list(path="C:/Users/tester/Desktop/Strelnikov/noise1"):
    SCOlist = []
    files = os.listdir(path)
    for filename in glob.glob(os.path.join(path, '*.wav')):
        wave = read_wave(filename)
        spect = wave.make_spectrum()
        node = spect.amps
        SCO = deviation(node,confidence=0.99)
        SCOlist.append(SCO)
    return SCOlist  

def calculate_CRFACT_list(path="C:/Users/tester/Desktop/Strelnikov/noise1"):
    CRFACTlist = []
    files = os.listdir(path)
    for filename in glob.glob(os.path.join(path, '*.wav')):
        wave = read_wave(filename)
        signal_level = rms_flat(wave.ys)
        peak_level = max(abs(wave.ys))
        CRFACT = peak_level/signal_level
        CRFACTlist.append(CRFACT)
    return CRFACTlist  

def calculate_CENTROID_list(path="C:/Users/tester/Desktop/Strelnikov/noise1"):
    CENTROIDlist = []
    files = os.listdir(path)
    for filename in glob.glob(os.path.join(path, '*.wav')):
        wave = read_wave(filename)
        CENTROID = wave.calculate_centroid()
        CENTROIDlist.append(CENTROID)
    return CENTROIDlist

def calculate_SPREAD_list(path="C:/Users/tester/Desktop/Strelnikov/noise1"):
    SPREADlist = []
    files = os.listdir(path)
    for filename in glob.glob(os.path.join(path, '*.wav')):
        wave = read_wave(filename)
        SPREAD = wave.calculate_ss()
        SPREADlist.append(SPREAD)
    return SPREADlist

def calculate_FLATNESS_list(path="C:/Users/tester/Desktop/Strelnikov/noise1"):
    FLATNESSlist = []
    files = os.listdir(path)
    for filename in glob.glob(os.path.join(path, '*.wav')):
        wave = read_wave(filename)
        FLATNESS = wave.calculate_sf()
        FLATNESSlist.append(FLATNESS)
    return FLATNESSlist

def load_list_of_wav(path="C:/Users/tester/Desktop/Strelnikov/noise1"):
    audiolist = []
    files = os.listdir(path)

    for filename in glob.glob(os.path.join(path, '*.wav')):
        wave = read_wave(filename)
        audiolist.append(wave)
    return audiolist

def calculate_middle_wave(audiolist):
    for i in range(len(audiolist)):
        if i == 0:
            average = audiolist[0]
        else:
            average += audiolist[i]
    return average

# Classifier processing
def evaluate_model(predictions, probs, train_predictions, train_probs, test_labels, train_labels):
    """Compare machine learning model to baseline performance.
    Computes statistics and shows ROC curve."""
    
    baseline = {}
    
    baseline['recall'] = recall_score(test_labels, [1 for _ in range(len(test_labels))])
    baseline['precision'] = precision_score(test_labels, [1 for _ in range(len(test_labels))])
    baseline['roc'] = 0.5
    
    results = {}
    
    results['recall'] = recall_score(test_labels, predictions)
    results['precision'] = precision_score(test_labels, predictions)
    results['roc'] = roc_auc_score(test_labels, probs)
    
    train_results = {}
    train_results['recall'] = recall_score(train_labels, train_predictions)
    train_results['precision'] = precision_score(train_labels, train_predictions)
    train_results['roc'] = roc_auc_score(train_labels, train_probs)
    
    for metric in ['recall', 'precision', 'roc']:
        print(f'{metric.capitalize()} Baseline: {round(baseline[metric], 2)} Test: {round(results[metric], 2)} Train: {round(train_results[metric], 2)}')
    
    # Calculate false positive rates and true positive rates
    base_fpr, base_tpr, _ = roc_curve(test_labels, [1 for _ in range(len(test_labels))])
    model_fpr, model_tpr, _ = roc_curve(test_labels, probs)

    plt.figure(figsize = (8, 6))
    plt.rcParams['font.size'] = 16
    
    # Plot both curves
    plt.plot(base_fpr, base_tpr, 'b', label = 'baseline')
    plt.plot(model_fpr, model_tpr, 'r', label = 'model')
    plt.legend();
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('ROC Curves');
    
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Oranges):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.figure(figsize = (10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size = 24)
    plt.colorbar(aspect=4)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size = 14)
    plt.yticks(tick_marks, classes, size = 14)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    
    # Labeling the plot
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize = 20,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        
    plt.grid(None)
    plt.tight_layout()
    plt.ylabel('True label', size = 18)
    plt.xlabel('Predicted label', size = 18)
    
#Model testing
def extract_data(signalPath):
    # This is signal data extracting
    signalMaxAmp = calculate_mean_amplitude_list(signalPath)
    #signalMaxAmpSplit = split_calculate_mean_amplitude_list(signalPath)
    signalSco = calculate_SCO_amplitude_list(signalPath)
    signalCrestFactor = calculate_CRFACT_list(signalPath)
    signalCentroid = calculate_CENTROID_list(signalPath)
    signalSpread = calculate_SPREAD_list(signalPath)
    signalFlatness = calculate_FLATNESS_list(signalPath)
    # Extracting MFCC params (in Dataframe)
    #signalMfcc = smfccp.compute_mfcc(signalPath)
    
    #sigLst1 = [item[0] for item in signalMaxAmpSplit]
    #sigLst2 = [item[1] for item in signalMaxAmpSplit]
    #sigLst3 = [item[2] for item in signalMaxAmpSplit]
    #sigLst4 = [item[3] for item in signalMaxAmpSplit]
    
    #signalDataFrame = {'Total MaxAmp': signalMaxAmp, 'Split MaxAmp 1': sigLst1, 'Split MaxAmp 2': sigLst2, 'Split MaxAmp 3': sigLst3, 'Split MaxAmp 4': sigLst4, 'SCO': signalSco, 'Crest Factor': signalCrestFactor, 'Centroid': signalCentroid, 'Spread': signalSpread, 'Flatness': signalFlatness}
    signalDataFrame = {'Total MaxAmp': signalMaxAmp, 'SCO': signalSco, 'Crest Factor': signalCrestFactor, 'Centroid': signalCentroid, 'Spread': signalSpread, 'Flatness': signalFlatness}
    
    signalDataFrame = pd.DataFrame(data=signalDataFrame)
    #signalDataFrame = pd.concat([signalDataFrame,signalMfcc], axis=1)
    
    return signalDataFrame

def plot_all_spectrogramms(audiolist, dataframe, resolution):
    i = 0
    dd = dataframe.loc[dataframe['Flag'] == 1]
    while (i != (len(dd))):
        print ("Signal №", dd.iloc[i].name)
        audiolist[dd.iloc[i].name].plot_spectrogram(audiolist[dd.iloc[i].name], resolution)
        i = i + 1