import torch
import numpy as np
import scipy.signal as sp


def frft(x,a):
    frft_x = torch.zeros_like(x, dtype=torch.complex64)
    for i in range(x.shape[0]):
        f = x[i, :]
        N = f.shape[-1]
        sN = torch.sqrt(torch.tensor(N))
        a = torch.remainder(torch.from_numpy(np.asarray(a)), 4.0)
        if a == 0.0:
            frft_x[i, :] = f
            continue
        if a == 2.0:
            frft_x[i, :] = torch.fliplr(f)
            continue
        if a == 1.0:
            ret = torch.fft.fft(f) / sN
            frft_x[i, :] = ret
            continue
        if a == 3.0:
            ret = torch.fft.ifft(f) * sN
            frft_x[i, :] = ret
            continue

        if a > 2.0:
            a = a - 2.0
            f = torch.fliplr(f)
        if a > 1.5:
            a = a - 1
            f = torch.fft.fft(f) / sN
        if a < 0.5:
            a = a + 1
            f = torch.fft.ifft(f) / sN
        alpha = a * np.pi / 2
        tana2 = torch.tan(alpha / 2)
        sina = torch.sin(alpha)
        f = torch.hstack((torch.zeros(f.shape[0], N-1), torch.from_numpy(sincinterp(f)), torch.zeros(f.shape[0], N-1))).to('cuda')
        chrp = torch.exp(-1j * np.pi / N * tana2 / 4 * torch.arange(-2 * N + 2, 2 * N - 1) ** 2).unsqueeze(0).repeat(f.shape[0],1).to('cuda')
        f = chrp * f
        c = np.pi / N / sina / 4
        ret = sp.fftconvolve(np.repeat(np.expand_dims(np.exp(1j * c.numpy() * np.arange(-(4 * N - 4), 4 * N - 3) ** 2),0), f.shape[0], axis=0),
                             f.cpu().numpy(),axes=-1)
        ret = torch.from_numpy(ret[:,4 * N - 4:8 * N - 7]).to('cuda') * np.sqrt(c / np.pi)
        ret = chrp * ret
        ret = torch.exp(-1j * (1 - a) * np.pi / 4) * ret[:,N - 1:-N + 1:2]
        frft_x[i, :] = ret
    if a==0. or a==2.:
        return frft_x.to('cuda')
    else:
        frft_x = torch.stack([frft_x.real, frft_x.imag],dim=-1)
        return frft_x.to('cuda')


def ifrft(f,a):
    return frft(f, -a)


def sincinterp(x):
    x = np.squeeze(x.cpu().detach().numpy())
    N = x.shape[1]
    y = np.zeros([x.shape[0],2 * N-1])
    y[:,:2 * N:2] = x
    xint = sp.fftconvolve(
        np.squeeze(y[:,:2 * N]),
        np.repeat(np.expand_dims(np.sinc(np.arange(-(2 * N - 3), (2 * N - 2)).T / 2),0),x.shape[0],axis=0),
    axes=-1)
    xint = xint[:,2 * N - 3: -2 * N + 3]
    return xint

