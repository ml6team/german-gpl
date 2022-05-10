from torch.utils.data import Dataset
import torch
import logging
from datasets import load_dataset
import logging
import pandas as pd
import torch
from tqdm import tqdm
from google.cloud import bigquery
import numpy as np
from src.config import dataset_params


def get_data(tokenizer, 
            max_source_text_len, 
            max_target_text_len,
            query_column_name,
            positive_column_name,
            negative_column_name,
            seed, 
            dataset_size, ):

    data = download_mmarco()
    df_train, df_val = create_dataframe(data, query_column_name, positive_column_name, negative_column_name, dataset_size=dataset_size)

    print(f"FULL Dataset: {df_train.shape}")

    train_data = T5InputDataset(
        df_train, 
        tokenizer, 
        max_source_text_len, 
        max_target_text_len,
        positive_column_name,
        query_column_name,  
    )

    # not used during training at all
    val_data = T5InputDataset(
        df_val, 
        tokenizer, 
        max_source_text_len, 
        max_target_text_len,
        positive_column_name,
        query_column_name,
    )

    return train_data, val_data

def download_mmarco():
    """
    Downlaod mmarco dataset provided by huggingface.co
    """
    logging.info('Start initialisation of mmarco dataset.')
    dataset = load_dataset('unicamp-dl/mmarco', dataset_params['LANGUAGE'], cache_dir='../dataset')
    return dataset

def create_dataframe(dataset, query_column, positive_column, negative_column, dataset_size=100):
    '''
    Create query_doc dataframe based on the mmarco dataset. 

    Args:
        - dataset: mmarco dataset
    
    Return:
        - dataframe: Query, Positive_Statement
    '''
    train = dataset['train'][:dataset_size]

    d = {}
    i = 0
    for i in tqdm(range(0, len(train['query']))):
        query = train['query'][i]
        positive = train['positive'][i]
        negative = train['negative'][i]

        d[i] = {
            query_column: query,
            positive_column: positive,
            negative_column: negative
        }

        i = i + 1

    df = pd.DataFrame.from_dict(d, 'index')
    # split dataframe
    msk = np.random.rand(len(df)) < 0.8
    train = df[msk]
    test = df[~msk]
    return train, test


class T5InputDataset(Dataset):
    """
    Creating a MMarcoQueryDoc dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model
    """

    def __init__(
        self, dataframe, tokenizer, source_len, target_len, source_text, target_text
    ):
        """
        Initializes a Dataset class

        Args:
            dataframe (pandas.DataFrame): Input dataframe
            tokenizer (transformers.tokenizer): Transformers tokenizer
            source_len (int): Max length of source text
            target_len (int): Max length of target text
            source_text (str): column name of source text
            target_text (str): column name of target text
        """
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = target_len
        self.target_text = self.data[target_text]
        self.source_text = self.data[source_text]

    def __len__(self):
        """returns the length of dataframe"""

        return len(self.target_text)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""

        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])

        # cleaning data so as to ensure data is in string type
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.summ_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            "target_ids_y": target_ids.to(dtype=torch.long),
        }

