import numpy as np
import librosa
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import PolynomialFeatures as PF
from sklearn.linear_model import LinearRegression as LR
from sklearn.pipeline import make_pipeline
import time

sstr = str(time.time()).split('.')[0]
""" steps
1.get hz of harmonic (?)
    sr=44100, 6s, error 1e-3 -> len < 1e3 when getting phi?
2.get phi's of each harmonic:
    irrelevant of amp, use whole process to fft, length need to be int times of 1/hz(harmonic), may have noise, need clustering to determine
3.stft
4.get amp-x function of each harmonic
    given time and sr, relevant to both phi and hz, use calcculated data from above to get baseline when amp == 1, then linear transform
5. generate voice
"""
# def read_audio(filename):
#     with open(filename, 'rb') as f:
#         x = pickle.load(f)
#         sr = int(filename.split('_')[0].split('/')[-1])
#     return x, sr

def get_freq(x, sr, maxh=15):
    fx = np.array(fft(x))
    fq = np.linspace(0, sr, len(fx))
    fq, fx = fq[0: len(fq) // 2], fx[0: len(fx) // 2]
    l1fx = np.abs(fx)
    freq = []
    argm = l1fx.argmax()
    freq.append(fq[argm])
    for i in range(1, maxh):
        hhzidx = argm * (i + 1)
        aa = hhzidx - 5 if hhzidx >= 5 else 0
        bb = hhzidx + 5 if hhzidx + 5 < len(l1fx) else len(l1fx) - 1
        if aa >= len(l1fx):
            break
        idx = l1fx[aa:bb].argmax()
        freq.append(fq[idx + aa])
    return np.array(freq)

def get_phase(x, sr, freq):
    t = np.linspace(0, len(x) / sr, len(x))
    phase = []
    for i in range(1, len(freq) + 1):
        tmp_phase, diff = [], []
        for j in range(len(t) - 1, len(t) // 5 * 4, -1): # --
            tmp = t[j] * freq[i - 1]
            sub = np.abs(round(tmp) - tmp)
            if sub < 1e-3:
                ff = fft(x[:j])
                mm = np.abs(ff)[:len(ff) // 2].argmax()
                diff.append(sub)
                tmp_phase.append(np.angle(ff[mm]))
        tmp_phase, diff = np.array(tmp_phase), np.array(diff)
        if (tmp_phase.max() - tmp_phase.min()) / tmp_phase.max() < 0.1: # --
            phase.append(tmp_phase[diff.argmin()])
        else:
            centre, clusters = [], []
            
            for it in tmp_phase:
                flag = False
                for j in range(len(centre)):
                    if np.abs((it-centre[j])/centre[j]) < 0.30: # ---
                        clusters[j].append(it)
                        centre[j] = np.array(clusters[j]).mean()
                        flag = True
                        break
                if not flag:
                    clusters.append([it,])
                    centre.append(it)
            for j in range(len(clusters)):
                clusters[j] = len(clusters[j])
            phase.append(centre[np.array(clusters).argmax()])
    return np.array(phase)

def stft(x, sr, dt=0.05, max_hz=None):
    if max_hz is None:
        max_hz = sr // 2
    o = int(sr * dt)
    c = [x[i:i + o] for i in range(0, len(x), o)]
    c.pop()

    fft_c = [np.array(fft(_)[:int(max_hz / sr * o)]) for _ in c]
    fftc0 = fft_c[0].copy()
    fft_c = np.transpose(np.array(fft_c))
    time = np.linspace(0, len(x) / sr, len(c))
    hz = np.linspace(0, fft_c.shape[0] * sr / o, fft_c.shape[0])

    # ansamp = np.abs(fft_c)
    # fq=hz
    # tt = time
    # plt.figure()
    # plt.contourf(tt, fq[0: len(fq) // 2], ansamp[0: len(fq) // 2, :])
    # argm = ansamp.argmax()
    # plt.colorbar()
    # print(tt[argm % ansamp.shape[1]], fq[argm // ansamp.shape[1]])
    # plt.scatter(tt[argm % ansamp.shape[1]], fq[argm // ansamp.shape[1]], color='red')
    # plt.show()
    # exit()
    return time, hz, fft_c, fftc0

def get_amp(fq, farr, sr, phase, freq):
    ansamp = np.abs(farr)
    hzamp = []
    for i in range(1, len(phase) + 1):
        if i > len(freq):
            break
        for j in range(1, len(fq)):
            if fq[j-1] < freq[i-1] < fq[j]:
                idx = j - 1
                break
        if idx - 3 >= farr.shape[0]:
            break
        bb = idx + 3 if idx + 3 < farr.shape[0] else farr.shape[0] - 1
        aa = idx - 3 if idx - 3 >= 0 else 0
        tmp = ansamp[aa: bb, :]

        argm = tmp.max(axis=1).argmax() + aa
        t = np.linspace(0, farr.shape[0] / sr, farr.shape[0]) # ????
        cc = np.cos(2 * np.pi * freq[i-1] * t + phase[i-1])
        line = np.abs(fft(cc)).max()
        # line = 1
        hzamp.append(tmp.max(axis=0) / line)
        # print(hzamp[-1].max()) # ---------------
    return np.array(hzamp)

def reg_amp(tt, fftamp, saveplt=False, figname='newregamp.png'):
    model,e1lst = [], []
    if saveplt:
        plt.figure(figsize=(9, 9))
    for idx in range(len(phase)):
        t0, hz0 = tt, fftamp[idx] + 1e-10 # -----------
        length = len(t0) // 6 # -----------------------------
        lr = LR()
        # print(hz0.min())
        # input()
        lr.fit(t0[0: length].reshape(-1, 1), np.log(hz0[0:length]).reshape(-1, 1))
        e1lst.append(lr.coef_[0, -1])
        poly = 5
        while True:
            pl = make_pipeline(PF(poly), LR()) 
            pl.fit(t0.reshape(-1, 1), np.log(hz0).reshape(-1, 1))
            if pl[1].coef_[0,-1] < 0:
                break
            if poly > 11:
                pl = LR()
                pl.fit(t0.reshape(-1, 1), np.log(hz0).reshape(-1, 1))
                break
            poly += 2
        model.append(pl)
        if saveplt: # ------
            pred = np.exp(model[idx].predict(t0.reshape(-1, 1))).reshape(-1)
            plt.subplot(330 + idx + 1)
            plt.plot(t0, hz0)
            plt.plot(t0, pred)
            plt.title(f'{idx}th harmonic' if idx != 0 else f'base')
    if saveplt:
        plt.savefig(sstr[5:] + figname)
    return model, e1lst

def sim(hz, tau, t0, tmax, sr, A=1., phi=0.):
    if abs(A) < 1e-8:
        return np.zeros(int(tmax * sr))
    om = 2 * np.pi * hz
    sim = [np.cos(om * i / sr + phi) for i in range(int(tmax * sr))]
    sim = np.array(sim)
    for i in range(len(sim)):
        if i / sr < t0:
            sim[i] = 0
    return sim * A

def generate_voice(phase, model, hz, sr, tmax):
    voice = sim(0, 0, 0, tmax=tmax, sr=sr, A=0.)
    for i in range(len(phase)):
        tmp = sim((i+1) * hz, 0, t0=0., tmax=tmax, sr=sr, phi=phase[i])
        tt = np.linspace(0, tmax, len(tmp))
        amp = np.exp(model[i].predict(tt.reshape(-1, 1))).reshape(-1)
        tmp = amp * tmp
        voice += tmp
    voice = voice * 0.4 / voice.max()
    return voice

def write_mp3(name, x, sr, fmt='wav'):
    import soundfile as sf
    sf.write(name, x, sr, format=fmt)
    print(f'written {name} sucessfully')

x, sr = librosa.load('./data/10.mp3', sr=12000)
x = x[int(5000 * sr / 8000): ] # -------------------------

freq = get_freq(x, sr)
print('freq done', len(freq), np.round(freq))
phase = get_phase(x, sr, freq)
print('phase done', len(phase))
tt, hz, farr, farr0 = stft(x, sr, max_hz=sr // 2)
print('tt', len(tt), 'hz',len(hz), 'farr', farr.shape)
amp_t = get_amp(hz, farr, sr, phase, freq)
print('amp done', amp_t.shape)
model, _ = reg_amp(tt, amp_t, saveplt=False)
print('reg done')
voice = generate_voice(phase, model, hz=288, sr=sr, tmax=6.)
print('generate done')
write_mp3(f'new{sstr[5:]}test288.wav', voice, sr=sr, fmt='wav')





