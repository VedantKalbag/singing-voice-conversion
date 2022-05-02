import pandas as pd
import os
import numpy as np
import librosa
import pyworld
from tqdm import tqdm
import multiprocessing

def extract_world(rootDir):
    dirName, subdirList, _ = next(os.walk(rootDir))
    
    for subdir in tqdm(subdirList):
        _, _, fileList = next(os.walk(os.path.join(dirName,subdir)))
        
        # embedding = np.load(f'../embeddings/{subdir}.npy')
        # rootdirs = []
        # subdirs=[]
        filenames=[]
        pathnames = []
        # audio_files=[]
        embs=[]
        sp_s = []
        # f0_s = []
        # ap_s = []
        # print(subdir)
        # print(os.getcwd())
        # print((os.path.join(os.getcwd(),subdir)+'.pkl'))
        # print(!os.path.exists(os.path.join(os.getcwd(),subdir)+'.pkl'))
        # if not (os.path.exists(os.path.join(os.getcwd(),subdir)+'.pkl')):
            # print("New subdir")
        if not os.path.exists(f'{subdir}.pkl'):
            for filename in fileList:
                df=pd.DataFrame()
                # rootdirs.append(rootDir)
                # subdirs.append(subdir)
                filenames.append(filename)
                path = os.path.join(dirName, subdir, filename)
                pathnames.append(path)
                wav, _ = librosa.load(path, sr=44100, mono=True) # TODO: consider changing sample rates
                # print(wav.shape)
                # break
                # audio_files.append(wav)
                wav = wav.astype(np.float64)   
                f0, timeaxis = pyworld.harvest(
                wav, fs=44100, frame_period=5, f0_floor=71.0, f0_ceil=800.0) # frame_period of 5 implies 2048 block
                # f0_s.append(f0)
                # Finding Spectogram
                sp = pyworld.cheaptrick(wav, f0, timeaxis, fs=44100)
                coded_sp = pyworld.code_spectral_envelope(sp, 44100, 80)
                sp_s.append(coded_sp)
                # Finding aperiodicity
                # ap = pyworld.d4c(wav, f0, timeaxis, fs=44100)
                # ap_s.append(ap)
                # embs.append(embedding)
            df['filename'] = filenames
            df['pathname'] = pathnames   
            # df['f0'] = f0_s
            df['sp'] = sp_s
            # df['ap'] = sp_s
            # df['embedding'] = embs
            df.to_pickle(f'{subdir}.pkl')
            print(f"Saved {subdir}")
        else:
            print(f"Skipping {subdir}")

        
if __name__=='__main__':
    extract_world('../../../DAMP-multi/processed_data/cropped_input_10s')