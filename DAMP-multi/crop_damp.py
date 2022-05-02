import librosa
import os
import numpy as np
import soundfile as sf
from tqdm import tqdm

np.random.seed(42)
rootDir = './processed_data/cropped_input_data'
dirName, subdirList, _ = next(os.walk(rootDir))
for subdir in tqdm(subdirList):
    print(subdir)
    _, _, fileList = next(os.walk(os.path.join(dirName,subdir)))
    if not os.path.isdir(os.path.join(dirName,'../processed_data/cropped_input_10s/', subdir)):
        os.mkdir(os.path.join(dirName,'../processed_data/cropped_input_10s/', subdir))
    for filename in fileList:
        try:
            if not os.path.exists(os.path.join(dirName,f'./processed_data/cropped_input_10s/{subdir}/{filename[:-4]}.wav')):
                path = os.path.join(dirName, subdir, filename)
                wav, sr = librosa.load(path, mono=True)
                left = np.random.randint(wav.shape[0]/4,wav.shape[0]/2)
                uttr = wav[left:left+(44100*30)]
                
                sf.write(os.path.join(dirName,f'./processed_data/cropped_input_10s/{subdir}/{filename[:-4]}.wav'),uttr,sr)
        except:
            pass
