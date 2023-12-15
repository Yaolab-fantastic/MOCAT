import torch
import warnings
warnings.filterwarnings('ignore')
from utils import train_test
if __name__ == "__main__":

    data_dir = ''
    result_dir = ''
    batch_size = 36

    with torch.cuda.device(0):
        train_test(data_dir, result_dir, batch_size)
