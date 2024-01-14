import torch
from torch.utils.data import Dataset
import os
import numpy as np
import random
import scipy.signal as sp
from scipy.signal import butter, lfilter
from data_aug import data_transform
import copy


def frft(f, a):
    """
    Calculate the fast fractional fourier transform.
    Parameters
    ----------
    f : numpy array
        The signal to be transformed.
    a : float
        fractional power
    Returns
    -------
    data : numpy array
        The transformed signal.
    References
    ---------
     .. [1] This algorithm implements `frft.m` from
        https://nalag.cs.kuleuven.be/research/software/FRFT/
    """
    ret = np.zeros_like(f, dtype=np.complex)
    f = f.copy().astype(np.complex)
    N = len(f)
    shft = np.fmod(np.arange(N) + np.fix(N / 2), N).astype(int)
    sN = np.sqrt(N)
    a = np.remainder(a, 4.0)

    # Special cases
    if a == 0.0:
        return f
    if a == 2.0:
        return np.flipud(f)
    if a == 1.0:
        ret[shft] = np.fft.fft(f[shft]) / sN
        return ret
    if a == 3.0:
        ret[shft] = np.fft.ifft(f[shft]) * sN
        return ret

    # reduce to interval 0.5 < a < 1.5
    if a > 2.0:
        a = a - 2.0
        f = np.flipud(f)
    if a > 1.5:
        a = a - 1
        f[shft] = np.fft.fft(f[shft]) / sN
    if a < 0.5:
        a = a + 1
        f[shft] = np.fft.ifft(f[shft]) * sN

    # the general case for 0.5 < a < 1.5
    alpha = a * np.pi / 2
    tana2 = np.tan(alpha / 2)
    sina = np.sin(alpha)
    f = np.hstack((np.zeros(N - 1), sincinterp(f), np.zeros(N - 1))).T

    # chirp premultiplication
    chrp = np.exp(-1j * np.pi / N * tana2 / 4 *
                     np.arange(-2 * N + 2, 2 * N - 1).T ** 2)
    f = chrp * f

    # chirp convolution
    c = np.pi / N / sina / 4
    ret = sp.fftconvolve(
        np.exp(1j * c * np.arange(-(4 * N - 4), 4 * N - 3).T ** 2),
        f
    )
    ret = ret[4 * N - 4:8 * N - 7] * np.sqrt(c / np.pi)

    # chirp post multiplication
    ret = chrp * ret

    # normalizing constant
    ret = np.exp(-1j * (1 - a) * np.pi / 4) * ret[N - 1:-N + 1:2]

    return ret


def ifrft(f, a):
    """
    Calculate the inverse fast fractional fourier transform.
    Parameters
    ----------
    f : numpy array
        The signal to be transformed.
    a : float
        fractional power
    Returns
    -------
    data : numpy array
        The transformed signal.
    """
    return frft(f, -a)


def sincinterp(x):
    N = len(x)
    y = np.zeros(2 * N - 1, dtype=x.dtype)
    y[:2 * N:2] = x
    xint = sp.fftconvolve(
        y[:2 * N],
        np.sinc(np.arange(-(2 * N - 3), (2 * N - 2)).T / 2),
    )
    return xint[2 * N - 3: -2 * N + 3]


def butter_bandpass_filter(data, lowcut=8, highcut=30, samplingRate=512, order=4):
    y = np.zeros_like(data).astype(np.float32)
    nyq = 0.5 * samplingRate
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    for i in range(data.shape[0]):
        y[i, :] = lfilter(b, a, data[i, :])
    return y


def iir_bandpass_filter(data, lowcut=8, highcut=30, samplingRate=512, order=4):
    y = np.zeros_like(data).astype(np.float32)
    nyq = 0.5 * samplingRate
    low = lowcut / nyq
    high = highcut / nyq
    b, a = sp.iirfilter(order, [low, high], btype='band')
    for i in range(data.shape[0]):
        y[i, :] = sp.filtfilt(b, a, data[i, :])
    return y


class DatasetProcessing(Dataset):
    def __init__(self, data_path, phase, window_size, mask_ratio, win_train=1, down_sample=1, sample_freq=1000, transform=None, device='cuda'):
        self.bci_data_path = os.path.join(data_path, phase) #data/train data/test
        self.transform = transform
        self.bci_file_name = []
        self.sample_freq = int(sample_freq/down_sample)
        self.win_train = int(self.sample_freq*win_train)
        self.phase = phase
        self.down_sample = down_sample
        self.label = []
        self.device = device
        class_num = 0.
        self.class_dict = {}
        for class_name in os.listdir(self.bci_data_path):
            class_bci_file = os.listdir(os.path.join(self.bci_data_path, class_name))
            self.bci_file_name.extend(class_bci_file)
            self.label.extend([class_num]*len(class_bci_file))
            self.class_dict[class_num] = class_name
            class_num += 1.
        self.label = np.array(self.label).astype(np.float32)
        self.height, self.width = window_size

        self.num_patches = self.height * self.width
        self.num_mask = int(mask_ratio * self.num_patches)

    def __getitem__(self, index):
        def simple_batch_norm_1d(x, dim):
            eps = 1e-5
            x_mean = torch.mean(x, dim=dim, keepdim=True)
            x_var = torch.mean((x - x_mean) ** 2, dim=dim, keepdim=True)
            x_hat = (x - x_mean) / (torch.sqrt(x_var) + eps)
            return x_hat

        label_name = self.class_dict[self.label[index]]
        time_start = random.randint(35, int(self.sample_freq * 4 + 35 - self.win_train))
        x1 = time_start
        x2 = time_start + self.win_train
        bci_data = np.load(os.path.join(self.bci_data_path, label_name, self.bci_file_name[index]))[:,
                   ::self.down_sample][:, x1:x2].astype(np.float32)
        bci_data = iir_bandpass_filter(bci_data, 3, 50, self.sample_freq, 4).astype(np.float32)
        ori_bci_data = copy.deepcopy(bci_data)
        bci_data = np.ascontiguousarray(data_transform(bci_data)).astype(np.float32)
        # bci_data = np.pad(bci_data,((1, 1), (3, 3)),'constant')
        bci_data = torch.from_numpy(bci_data).to(self.device)
        ori_bci_data = torch.from_numpy(ori_bci_data).to(self.device)
        if self.transform is None:
            bci_data = simple_batch_norm_1d(bci_data, dim=0)
            ori_bci_data = simple_batch_norm_1d(ori_bci_data, dim=0)
        bci_data = torch.unsqueeze(bci_data, 0)
        ori_bci_data = torch.unsqueeze(ori_bci_data, 0)

        mask = np.hstack([np.zeros(self.num_patches - self.num_mask), np.ones(self.num_mask),])
        np.random.shuffle(mask)
        mask = torch.from_numpy(mask).to(self.device)
        return ori_bci_data, bci_data, mask

    def __len__(self):
        return len(self.bci_file_name)


def data_generator_np(data_path, batch_size, window_size, mask_ratio, win_train=1, down_sample=1, sample_freq=1000, device='cuda'):
    train_dataset = DatasetProcessing(data_path, 'train', window_size, mask_ratio, win_train=win_train, down_sample=down_sample, sample_freq=sample_freq, device=device)
    test_dataset = DatasetProcessing(data_path, 'test', window_size, mask_ratio, win_train=win_train, down_sample=down_sample, sample_freq=sample_freq, device=device)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=False,
                                               num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              drop_last=False,
                                              num_workers=0)

    return train_loader, test_loader
