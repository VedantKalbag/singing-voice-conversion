import pandas as pd
import librosa
import pyworld
import numpy as np

from torch.utils import data
import torch
import numpy as np
import pickle 
import os  
import pyworld  
import pandas as pd

from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from multiprocessing import Process, Manager   


class SpeechVocalDataset(data.Dataset):
    """Dataset class for the SpeechVocalDataset dataset."""
    def __init__(self, root_dir, len_crop, dataset='train'):
        """Initialize and preprocess the SpeechVocalDataset dataset."""
        # lookup_path = root_dir + '/../input_data/input_file_list.csv'
        self.root_dir = root_dir
        self.len_crop = len_crop
        self.step = 10
        # self.lookup = pd.read_csv(lookup_path, header=0)
        # self.lookup = self.lookup.sample(frac=1) # randomly shuffle the dataset
        if dataset == 'train':
            self.dataset = pd.read_pickle(f'{self.root_dir}/train.pkl')
        if dataset == 'valid':
            self.dataset = pd.read_pickle(f'{self.root_dir}/valid.pkl')
        if dataset == 'test':
            self.dataset = pd.read_pickle(f'{self.root_dir}/test.pkl')
        # self.valid_dataset = pd.read_pickle(f'{self.root_dir}/valid.pkl')
        # self.dataset = self.dataset.sample(frac=1) # randomly shuffle the dataset
        # self.train_dataset = list(dataset)
        self.num_tokens = len(self.dataset)
        print('Finished loading the dataset...')
        
    def load_data(self, dataset, i):
        return dataset.iloc[i]['sp_coded'], dataset.iloc[i]['embedding']
        
    def __getitem__(self, index):
        dataset = self.dataset#lookup
        sp_tmp,emb = self.load_data(dataset,index)
        if sp_tmp.shape[0] < self.len_crop:
            len_pad = self.len_crop - sp_tmp.shape[0]
            sp = np.pad(sp_tmp, ((0,len_pad),(0,0)), 'constant')
        elif sp_tmp.shape[0] > self.len_crop:
            left = np.random.randint(sp_tmp.shape[0]-self.len_crop)
            sp = sp_tmp[left:left+self.len_crop, :]
        else:
            sp = sp_tmp

        return sp,emb
    def __len__(self):
        """Return the number of spkrs."""
        return self.num_tokens


class DAMPDataset(data.Dataset):
    """Dataset class for the SpeechVocalDataset dataset."""
    def __init__(self, root_dir, len_crop, dataset='train'):
        """Initialize and preprocess the SpeechVocalDataset dataset."""
        # lookup_path = root_dir + '/../input_data/input_file_list.csv'
        self.root_dir = root_dir
        self.len_crop = len_crop
        self.step = 10
        # self.lookup = pd.read_csv(lookup_path, header=0)
        # self.lookup = self.lookup.sample(frac=1) # randomly shuffle the dataset
        if dataset == 'train':
            self.dataset = pd.read_csv(f'{self.root_dir}/train.csv')
        if dataset == 'valid':
            self.dataset = pd.read_csv(f'{self.root_dir}/valid.csv')
        if dataset == 'test':
            self.dataset = pd.read_csv(f'{self.root_dir}/test.csv')
        # self.valid_dataset = pd.read_pickle(f'{self.root_dir}/valid.pkl')
        # self.dataset = self.dataset.sample(frac=1) # randomly shuffle the dataset
        # self.train_dataset = list(dataset)
        self.num_tokens = len(self.dataset)
        print('Finished loading the dataset...')
        
    def load_data(self, dataset, i):
        filename = dataset['filename'][i]
        subdirname = dataset['subdir'][i]
        d=np.load(os.path.join(self.root_dir, 'pyworld', f'{subdirname}.pkl'), allow_pickle=True)

        sp_coded = d[d['filename'] == filename]['sp'].to_numpy()[0]
        emb = np.load(os.path.join(self.root_dir, 'embeddings', f'{subdirname}.npy'))
        return sp_coded, emb
        # return dataset.iloc[i]['sp_coded'], dataset.iloc[i]['embedding']
        
    def __getitem__(self, index):
        dataset = self.dataset#lookup
        sp_tmp,emb = self.load_data(dataset,index)
        if sp_tmp.shape[0] < self.len_crop:
            len_pad = self.len_crop - sp_tmp.shape[0]
            sp = np.pad(sp_tmp, ((0,len_pad),(0,0)), 'constant')
        elif sp_tmp.shape[0] > self.len_crop:
            left = np.random.randint(sp_tmp.shape[0]-self.len_crop)
            sp = sp_tmp[left:left+self.len_crop, :]
        else:
            sp = sp_tmp

        return sp,emb
    def __len__(self):
        """Return the number of spkrs."""
        return self.num_tokens
    
def get_loader(root_dir, batch_size=16, len_crop=128, num_workers=0, dataset='VCTK'):
    """Build and return a data loader."""
    if dataset == 'VCTK':
        train_dataset = SpeechVocalDataset(root_dir, len_crop, 'train')
        valid_dataset = SpeechVocalDataset(root_dir, len_crop, 'valid')
        test_dataset = SpeechVocalDataset(root_dir, len_crop, 'test')
    elif dataset == 'DAMP':
        train_dataset = DAMPDataset(root_dir, len_crop, 'train')
        valid_dataset = DAMPDataset(root_dir, len_crop, 'valid')
        test_dataset = DAMPDataset(root_dir, len_crop, 'test')
    # datasets = train_val_dataset(dataset, val_split=0.25)
    worker_init_fn = lambda x: np.random.seed((torch.initial_seed()) % (2**32))
    train_loader = data.DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  drop_last=True,
                                  worker_init_fn=worker_init_fn)

    valid_loader = data.DataLoader(dataset=valid_dataset,
                                  batch_size=len(valid_dataset),
                                  shuffle=True,
                                  num_workers=num_workers,
                                  drop_last=True,
                                  worker_init_fn=worker_init_fn)

    test_loader = data.DataLoader(dataset=test_dataset,
                                  batch_size=len(test_dataset),
                                  shuffle=True,
                                  num_workers=num_workers,
                                  drop_last=True,
                                  worker_init_fn=worker_init_fn)

    return train_loader, valid_loader, test_loader


# train_set, val_set = torch.utils.data.random_split(dataset, [50000, 10000])
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-lambda_cd', help="lambda_cd", type=int, required=False, default=1)
    parser.add_argument('-dim_neck', help="", type=int, required=False, default=32)
    parser.add_argument('-dim_emb', help="", type=int, required=False, default=256)
    parser.add_argument('-dim_pre', help="", type=int, required=False, default=512)
    parser.add_argument('-freq',help='',type=int, required=False, default=32)
    parser.add_argument('-data_dir',help='',type=str, required=False, default='./DAMP-multi/processed_data')
    parser.add_argument('-batch_size', help='', type=int, required=False, default=1)
    parser.add_argument('-num_iters',help='',type=int, required=False, default=100000) # 1000000
    parser.add_argument('-len_crop',help='',type=int,required=False,default=128)
    parser.add_argument('-log_step',help='',type=int,required=False,default=10)
    parser.add_argument('-load_ckpt_path', help='', type=str, required=False, default='')
    parser.add_argument('-suffix', help='', type=str, required=False, default='')
    parser.add_argument('-load', help='', type=bool, required=False, default=False)
    parser.add_argument('-lr', help='', type=float, required=False, default=0.0001)
    config = parser.parse_args()

    import os
    from solver_encoder import Solver
    # from data_loader import get_loader
    from torch.backends import cudnn

    cudnn.benchmark = True

    train_loader, valid_set, test_set = get_loader(config.data_dir,batch_size=16,len_crop=config.len_crop,num_workers=4, dataset='DAMP')
    
    solver = Solver(train_loader, valid_set, test_set, config)

    solver.train()

