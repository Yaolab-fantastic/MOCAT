import torch
import warnings
warnings.filterwarnings('ignore')
from utils import train_test
import os
if __name__ == "__main__":

    data_dir = 'ROSMAP'
    result_dir = 'ROSMAP'
    batch_size = 36
    train_test(data_dir, result_dir, batch_size)
