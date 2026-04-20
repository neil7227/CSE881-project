import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime
from scipy import signal
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# 讀檔案
def read_signals(main_folder):
    signal_dict = {}
    time_dict = {}
    fs_dict = {}

    subfolders = next(os.walk(main_folder))[1]

    utc_start_dict = {}
    for folder_name in subfolders:
        csv_path = f'{main_folder}/{folder_name}/EDA.csv'
        df = pd.read_csv(csv_path)
        utc_start_dict[folder_name] = df.columns.tolist()

    for folder_name in subfolders:
        folder_path = os.path.join(main_folder, folder_name)
        files = os.listdir(folder_path)

        signals = {}
        time_line = {}
        fs_signal = {}
        
        desired_files = ['EDA.csv', 'BVP.csv', 'HR.csv', 'TEMP.csv', 'tags.csv', 'ACC.csv']
   
        for file_name in files:
            file_path = os.path.join(folder_path, file_name)

            if file_name.endswith('.csv') and file_name in desired_files:
                if file_name == 'tags.csv':
                    try:
                        df = pd.read_csv(file_path, header=None)
                        tags_vector = create_df_array(df)
                        tags_UTC_vector = np.insert(tags_vector, 0, utc_start_dict[folder_name])
                        signal_array = time_abs_(tags_UTC_vector)
                        time_array = None
                        fs = None
                    except pd.errors.EmptyDataError:
                        signal_array = []
                        time_array = None
                        fs = None
                else:
                    df = pd.read_csv(file_path)
                    if df.empty:
                        signal_array = []
                        time_array = None
                        fs = None
                    else:
                        fs_row = df.iloc[0]        
                        fs = int(fs_row.iloc[0])
                        df = df.iloc[1:]           
                        signal_array = df.values
                        time_array = np.linspace(0, len(signal_array)/fs, len(signal_array))

                signal_name = file_name.split('.')[0]
                signals[signal_name] = signal_array
                time_line[signal_name] = time_array
                fs_signal[signal_name] = fs

        signal_dict[folder_name] = signals
        time_dict[folder_name] = time_line
        fs_dict[folder_name] = fs_signal

    return signal_dict, time_dict, fs_dict

def create_df_array(df):
    return df.values

def time_abs_(tags_vector):
    return tags_vector

# ========== 新增濾波器函數 ==========
def butter_lowpass_filter(data, cutoff, fs, order=4):
    """低通濾波器 - 保留低頻，去除高頻雜訊"""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def butter_highpass_filter(data, cutoff, fs, order=4):
    """高通濾波器 - 去除低頻漂移"""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = filtfilt(b, a, data)
    return y

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """帶通濾波器 - 保留特定頻率範圍"""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band', analog=False)
    y = filtfilt(b, a, data)
    return y