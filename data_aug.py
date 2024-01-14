import torch
import numpy as np
import random
from scipy import signal


def dc_shift(data, shift_range):
    shift_value = random.randint(shift_range[0],shift_range[1])
    data = data + shift_value
    return data


def add_noise(data, sigma_scale):
    sigma = np.random.rand()*sigma_scale
    noise = np.random.normal(0, sigma, data.shape).astype(dtype=np.float32)
    data += noise
    return data


def band_stop(data, freq_range, band_width, sample_freq):
    start_freq = random.randint(freq_range[0], freq_range[1]-5)
    end_freq = start_freq+band_width
    wn1 = 2 * start_freq / sample_freq
    wn2 = 2 * end_freq / sample_freq
    b, a = signal.butter(4, [wn1, wn2], 'bandstop')
    filtedData = signal.filtfilt(b, a, data)
    return filtedData


def amplitude_scale(data, scale_range):
    scale = np.random.uniform(low=scale_range[0], high=scale_range[1], size=1)
    data *= scale
    return data


def data_transform(data, shift_range=[-10, 10], noise_sigma=0.2, freq_range=[3, 50], band_width=5, sample_freq=250, amplitude_range=[0.5, 2]):
    data = dc_shift(data, shift_range)
    data = add_noise(data, noise_sigma)
    data = band_stop(data, freq_range, band_width, sample_freq)
    data = amplitude_scale(data, amplitude_range)
    return data