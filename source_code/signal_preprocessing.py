import os
import numpy as np
import scipy.signal as sig
import RigolWFM.wfm as wfm
from constants import *

r_Sxx = np.load("/content/drive/MyDrive/source_code/r_Sxx.npy")


def random_rat_batch_generator(input_dir, batch_size=256, n_batches = 1e3):

    batch_Sxx = []
    n_batch = 0
    meas = [os.path.join(input_dir, name, 'measurements') for name in os.listdir(input_dir)]
    all_meas = []
    for m in meas:
        all_meas += [os.path.join(m, name) for name in os.listdir(m)]
    np.random.shuffle(all_meas)
    i = 0
    while n_batch < n_batches:
        if i == len(all_meas):
            i = 0
        meas_file_dir = all_meas[i]
        i += 1
        
        batch_Sxx += get_random_filtered_intervals(meas_file_dir)
        while len(batch_Sxx)>=batch_size:
            
            yield (np.array(batch_Sxx[:batch_size]), np.array(batch_Sxx[:batch_size]))
            batch_Sxx = batch_Sxx[batch_size:]
            n_batch += 1

   

def get_random_filtered_intervals(file_path, ratio=0.95):
    try:
        time, signal, fp = open_wfm_file(file_path)
    except:
        return None
    f, t, Sxx = get_stft(time, signal, fp)
    all_Sxx = []
    Sxx_t = Sxx.transpose()
    
    i = 0
    k = 0
    t_length = t.shape[0]
    idxs = list(range(0, t_length, TIME_INTERVAL))[:-1]
    selected_idxs = np.random.choice(idxs, int(np.round(ratio*len(idxs))))
    for idx in selected_idxs:
        new_Sxx_t = []
        for i in range(idx, idx+TIME_INTERVAL ):
            s_ener = np.median(Sxx_t[i])
            if s_ener < ENERGY_THRESHOLD:
                new_Sxx_t.append(Sxx_t[i])
            else:
                
                new_Sxx_t.append(r_Sxx)
        Sxx_to_save = np.array(new_Sxx_t).transpose().reshape(INPUT_SHAPE)
        all_Sxx.append(Sxx_to_save)
    return all_Sxx
    
    
def filtered_intervals_generator_to_predict(file_path, batch_size=1):
    try:
        time, signal, fp = open_wfm_file(file_path)
        f, t, Sxx = get_stft(time, signal, fp)
    except:
        yield None
    new_t = t[:TIME_INTERVAL]
    yield f, new_t
    Sxx_t = Sxx.transpose()
    new_Sxx_t = []
    batch_Sxx = []
    i = 0
    k = 0
    n_batches = 0
    while i < t.shape[0]:
        s_ener = np.median(Sxx_t[i])
        if s_ener < ENERGY_THRESHOLD:
            new_Sxx_t.append(Sxx_t[i])
        else:
            new_Sxx_t.append(r_Sxx)
        
        if i == t.shape[0] - 1:
            n_shape = len(new_Sxx_t)
            n_to_add = TIME_INTERVAL - n_shape
            new_Sxx_t += [r_Sxx]*n_to_add
            Sxx_to_save = np.array(new_Sxx_t).transpose().reshape(INPUT_SHAPE)
            batch_Sxx.append(Sxx_to_save)
            if n_batches != 0:
                while len(batch_Sxx) < batch_size:
                    batch_Sxx.append(Sxx_to_save)
            yield np.array(batch_Sxx)
            break
        i += 1
        if len(new_Sxx_t) == TIME_INTERVAL:
            Sxx_to_save = np.array(new_Sxx_t).transpose().reshape(INPUT_SHAPE)
            batch_Sxx.append(Sxx_to_save)
            i -= TIME_INTERVAL // 2
            new_Sxx_t = new_Sxx_t[i:]
            if len(batch_Sxx) == batch_size:
                yield np.array(batch_Sxx)
                n_batches += 1
                batch_Sxx = []
                


def filtered_intervals_generator(file_path):
    try:
        time, signal, fp = open_wfm_file(file_path)
        f, t, Sxx = get_stft(time, signal, fp)
    except:
        yield None
    new_t = t[:TIME_INTERVAL]
    yield f, new_t
    Sxx_t = Sxx.transpose()
    new_Sxx_t = []
    i = 0
    k = 0
    while i < t.shape[0]:
        s_ener = np.median(Sxx_t[i])
        if s_ener < ENERGY_THRESHOLD:
            new_Sxx_t.append(Sxx_t[i])
        else:
            new_Sxx_t.append(r_Sxx)
        i += 1
        if len(new_Sxx_t) == TIME_INTERVAL:
            Sxx_to_save = np.array(new_Sxx_t).transpose().reshape(INPUT_SHAPE)
            yield Sxx_to_save
            i -= TIME_INTERVAL // 2
            new_Sxx_t = new_Sxx_t[i:]
    
    
def filtered_intervals_generator2(file_path):
    time, signal, fp = open_wfm_file(file_path)
    f, t, Sxx = get_stft(time, signal, fp)
    new_t = t[:TIME_INTERVAL]
    yield f, new_t
    Sxx_t = Sxx.transpose()
    new_Sxx_t = []
    i = 0
    k = 0
    while i < t.shape[0]:
        s_ener = np.median(Sxx_t[i])
        if s_ener < ENERGY_THRESHOLD:
            new_Sxx_t.append(Sxx_t[i])
        else:
            new_Sxx_t.append(r_Sxx)
        i += 1
        if len(new_Sxx_t) == TIME_INTERVAL:
            Sxx_to_save = np.array(new_Sxx_t).transpose().reshape(INPUT_SHAPE)
            yield Sxx_to_save
            i -= TIME_INTERVAL // 2
            new_Sxx_t = new_Sxx_t[i:]
    
def get_filtered_intervals(file_path):
    time, signal, fp = open_wfm_file(file_path)
    f, t, Sxx = get_stft(time, signal, fp)
    all_Sxx = []
    Sxx_t = Sxx.transpose()
    new_Sxx_t = []
    i = 0
    while i < t.shape[0]:
        s_ener = np.median(Sxx_t[i])
        if s_ener < ENERGY_THRESHOLD:
            new_Sxx_t.append(Sxx_t[i])
        else:
            new_Sxx_t.append(r_Sxx)
        i += 1
        if len(new_Sxx_t) == TIME_INTERVAL:
            Sxx_to_save = np.array(new_Sxx_t).transpose().reshape(INPUT_SHAPE)
            all_Sxx.append(Sxx_to_save)
            i -= TIME_INTERVAL // 2
            new_Sxx_t = new_Sxx_t[i:]
    return all_Sxx

def get_stft(time, signal, fp = None, nperseg=4096//2, std = 100):
    """
    :param time: time array
    :param signal: measuerement array
    :param nperseg: window weight
    :return:
    """
    
    wind = sig.get_window(('gaussian', std), nperseg)
    try:
        f, t, Sxx = sig.stft(signal, fs=fp, window=wind, nperseg=nperseg, return_onesided=True, noverlap=int(0.9*nperseg))
    except:
        return None, None, None
    
    if fp <100100:
        f_idx = 369
        new_Sxx = Sxx[f_idx:]
        new_f = f[f_idx:]
        new_Sxx = new_Sxx[::5]**2
        new_f = new_f[::5]
        
    elif fp == 200000.0:
        f_idx = 184
        new_Sxx = Sxx[f_idx:]
        new_f = f[f_idx:]
        new_Sxx = new_Sxx[::6]
        new_f = new_f[::6]
        new_Sxx = new_Sxx[9:]**2
        new_f = new_f[9:]

    else:
        f_idx = 211
        new_Sxx = Sxx[f_idx:]
        new_f = f[f_idx:]
        new_Sxx = new_Sxx[::6]
        new_f = new_f[::6]
        new_Sxx = new_Sxx[4:]**2
        new_f = new_f[4:]
    
    return new_f, t, np.abs(new_Sxx)
    
    
    

    
def open_wfm_file(file_path):
    try:
        scope = 'DS1052E'
        w = wfm.Wfm.from_file(file_path, scope)
    except:
        scope = 'DS1104Z'
        w = wfm.Wfm.from_file(file_path, scope)

    signal = w.channels[0].raw
    time = w.channels[0].times
    time = time + np.abs(np.min(time))
    dt = w.channels[0].seconds_per_point
    fp = np.round(1.0 / dt)
    return time, signal, fp
