import numpy as np
import pyworld
import scipy.signal
from tqdm import tqdm

def butter_highpass(cutoff, fs, order=5):
    from scipy import signal
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def pySTFT(x, fft_length=1024, hop_length=256):
    import numpy as np
    from scipy.signal import get_window
    x = np.pad(x, int(fft_length//2), mode='reflect')
    
    noverlap = fft_length - hop_length
    shape = x.shape[:-1]+((x.shape[-1]-noverlap)//hop_length, fft_length)
    strides = x.strides[:-1]+(hop_length*x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape,
                                             strides=strides)
    
    fft_window = get_window('hann', fft_length, fftbins=True)
    result = np.fft.rfft(fft_window * result, n=fft_length).T
    
    return np.abs(result)    
    
def get_mel_spec(x):
    import numpy as np
    from scipy import signal
    from librosa.filters import mel
    from numpy.random import RandomState
    prng = RandomState(42)
    mel_basis = mel(16000, 1024, fmin=90, fmax=7600, n_mels=80).T
    min_level = np.exp(-100 / 20 * np.log(10))
    b, a = butter_highpass(30, 16000, order=5)
    y = signal.filtfilt(b, a, x)
    # Ddd a little random noise for model roubstness
    wav = y * 0.96 + (prng.rand(y.shape[0])-0.5)*1e-06
    # Compute spect
    D = pySTFT(wav).T
    # Convert to mel and normalize
    D_mel = np.dot(D, mel_basis)
    D_db = 20 * np.log10(np.maximum(min_level, D_mel)) - 16
    S = np.clip((D_db + 100) / 100, 0, 1)   
    return S 

def get_embedding(C,x):
    import torch
    import numpy as np
    len_crop = 128
    left = np.random.randint(0, x.shape[0]-len_crop)
    melsp = torch.from_numpy(x[np.newaxis, left:left+len_crop, : ]).cuda()
    emb = C(melsp).detach().squeeze().cpu().numpy()
    return emb
    # print(emb)

def calculate_embeddings(rootDir, writeDir):
    # rootDir=''
    import numpy as np
    import os
    import torch
    import librosa
    from model_bl import D_VECTOR
    from collections import OrderedDict
    import pyworld
    C = D_VECTOR(dim_input=80, dim_cell=768, dim_emb=256).eval().cuda()
    c_checkpoint = torch.load('3000000-BL.ckpt')
    new_state_dict = OrderedDict()
    for key, val in c_checkpoint['model_b'].items():
        new_key = key[7:]
        new_state_dict[new_key] = val
    C.load_state_dict(new_state_dict)
    
    dirName, subdirList, _ = next(os.walk(rootDir))


    for subdir in tqdm(subdirList):
        if os.path.exists(os.path.join(dirName, f'../processed_data/embeddings/{subdir}.npy')):
            print(f"Skipping {subdir}")
            continue
        _, _, fileList = next(os.walk(os.path.join(dirName,subdir)))
        spkr_emb = []
        for filename in fileList:
            # LOAD AUDIO
            path = os.path.join(dirName, subdir, filename)
            wav, _ = librosa.load(path, sr=44100, mono=True) # TODO: consider changing sample rates
            wav = wav.astype(np.float64) 

            # GET MEL SPECTROGRAM
            S = get_mel_spec(wav)
            S = S.astype(np.float32)

            # GET EMBEDDINGS
            emb = get_embedding(C, S)
            # print(emb)
            spkr_emb.append(emb)
        # print(subdir)
        # print(spkr_emb)
        embedding = np.mean(np.array(spkr_emb), axis=0).astype(np.float32)
        np.save(os.path.join(writeDir, subdir), embedding, allow_pickle=False) 

def get_pyworld(wav):
    f0, timeaxis = pyworld.harvest(
    wav, fs=44100, frame_period=5, f0_floor=71.0, f0_ceil=800.0) # frame_period of 5 implies 2048 block
    # Finding Spectogram
    sp = pyworld.cheaptrick(wav, f0, timeaxis, fs=44100)
    # Finding aperiodicity
    ap = pyworld.d4c(wav, f0, timeaxis, fs=44100)

    return f0, sp, ap
if __name__ == '__main__':
    calculate_embeddings('./processed_data/cropped_input_10s','./processed_data/embeddings')