from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import pandas as pd
root_dir = './processed_data'
df = pd.read_pickle(f'{root_dir}/pyworld/train.pkl')
train, other = train_test_split(shuffle(df), test_size=0.3)
test, valid = train_test_split(other, test_size=0.5)
train.to_pickle('./processed_data/train.pkl')
test.to_pickle('./processed_data/test.pkl')
valid.to_pickle('./processed_data/valid.pkl')