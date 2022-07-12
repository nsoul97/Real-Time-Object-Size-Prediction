import torch as th
from data import Data
from model import Encoder

dataset = Data(
    dataset_path='/home/soul/Development/NCSR/Human action prediction from hand movement for human-robot collaboration/dataset',
    strategy='one-out',
    max_pad_length=100,
    seed=0)

train_set, test_set = dataset.get_train_test_datasets(0)

pretrain_model = Encoder(N=3,
                         heads=16,
                         d_input=3,
                         max_pad_length=100,
                         d_model=128,
                         d_ff=256,
                         dropout=0.1)

x = th.rand(2, 100, 3)
y = pretrain_model(x)
print(y.shape)