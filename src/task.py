# entrypoint for custom job
from model.t5 import run as t5_run_train
from model.crossencoder import run as cross_encoder_run_train
from data.dataset import store_in_bg
import wandb
import sys
from torch import cuda



def train_t5():
    print('[Run mT5 fine-tuning]')
    wandb.init(project="mt5-fine-tuning")
    t5_run_train()


def extract_data():
    """
    run data extraction of huggingface dataset. MMarco data will be stored in defined bq table. 
    """
    print('[Run MMarco Extraction]')
    store_in_bg()

def train_cross_encoder():
    print('[Run CrossEncoder Training]')
    wandb.init(project="cross-encoder-fine-tuning")
    cross_encoder_run_train()
    pass

def system_check():
    print('[DEBUG]')
    device = 'cuda' if cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    

if __name__ == "__main__":
    print('Start task ...')
    args = sys.argv
    # args[0] = current file
    # args[1] = function name
    # args[2:] = function args : (*unpacked)
    globals()[args[1]](*args[2:])
    
    