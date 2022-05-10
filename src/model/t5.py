'''
Models training loop
'''
import torch
from tqdm import tqdm
import pandas as pd
from torch.utils.data import DataLoader
import torch
import os
from tqdm import tqdm
from transformers import MT5ForConditionalGeneration, T5Tokenizer
import transformers
from data.dataset import get_data
from config import JOB_NAME, model_params, dataset_params, train_params, val_params, BUCKET_NAME
from utils import create_dir, upload_from_directory
from datetime import datetime
import wandb
import numpy as np

# Setting up the device for GPU usage
from torch import cuda

def init_model_from_path(model_path):
    """
    Initialize model based on given path

    Return:
        - (tokenizer, model)
    """
    tokenizer = T5Tokenizer.from_pretrained(model_path, local_files_only=True)
    model = MT5ForConditionalGeneration.from_pretrained(model_path, local_files_only=True)
    return tokenizer, model

def run():
    """
    Main entrypoint for the training. Retrieve the dataset, initilize the tokenizer and model, start the training itself.
    """
    device = 'cuda' if cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    
    # tokenzier for encoding the text
    tokenizer = T5Tokenizer.from_pretrained(model_params["MODEL"])
    model = MT5ForConditionalGeneration.from_pretrained(model_params["MODEL"])
    model = model.to(device)

    # get dataset
    train_data, val_data = get_data(tokenizer, 
        model_params['MAX_SOURCE_TEXT_LENGTH'], 
        model_params['MAX_TARGET_TEXT_LENGTH'],
        dataset_params['QUERY_COLUMN_NAME'], 
        dataset_params['POSITIVE_COLUMN_NAME'],
        dataset_params['NEGATIVE_COLUMN_NAME'],
        model_params['SEED'], 
        dataset_params['DATASET_SIZE'])


    # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
    training_loader = DataLoader(train_data, **train_params)
    val_loader = DataLoader(val_data, **val_params)

    # Defining the optimizer that will be used to tune the weights of the network in the training session.
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=model_params["LEARNING_RATE"]
    )

    


    print(f'[Start training on device: {device} ... ]')

    for epoch in range(model_params["TRAIN_EPOCHS"]):
        train(epoch, tokenizer, model, device, training_loader, optimizer, val_loader)
        save_model(model, tokenizer, step='final')

    # evaluating test dataset
    output_dir = './tmp'
    print(f"[Initiating Validation]...\n")
    for epoch in range(model_params["VAL_EPOCHS"]):
        predictions, actuals = validate(epoch, tokenizer, model, device, val_loader)
        final_df = pd.DataFrame({"Generated Text": predictions, "Actual Text": actuals})
        final_df.to_csv(os.path.join(output_dir, "predictions.csv"))

    # upload to bucket
    upload_from_directory('./tmp', BUCKET_NAME, JOB_NAME)
    print(f"[Upload to Bucket Completed.]\n")

    print(f"[Validation Completed.]\n")
    pass
    
def train(epoch, tokenizer, model, device, loader, optimizer, val_loader, validation_step=500, early_stopping_patient=3):
    
    """
    Function to be called for training with the parameters
    """

    best_val_loss = None
    early_stopping_counter = 0
    stop_training = False

    model.train()
    for _, data in enumerate(tqdm(loader), 0):
        if stop_training:
            break
        y = data["target_ids"].to(device, dtype=torch.long)
        #y_ids = y[:, :-1].contiguous()
        #lm_labels = y[:, 1:].clone().detach()
        #lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data["source_ids"].to(device, dtype=torch.long)
        mask = data["source_mask"].to(device, dtype=torch.long)

        outputs = model(
            input_ids=ids,
            attention_mask=mask,
            labels=y
        )
        loss = outputs[0]

        wandb.log({'loss': loss})

        if _ % 100 == 0:
            # todo logging
            print(f'epoch {str(epoch)} / iteration {_} /// loss {str(loss)}')

        if _ % 5000 == 0:
            save_model(model, tokenizer, _)
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if _ % validation_step == 0:
            # early stopping based on val los
            val_loss = model_validate(val_loader, device, model)
            wandb.log({'val_loss': val_loss})
            
            if best_val_loss is None:
                best_val_loss = val_loss

            if best_val_loss > val_loss:
                best_val_loss = val_loss

                # reset early stopping counter
                if early_stopping_counter > 0:
                    early_stopping_counter = 0

            elif val_loss > best_val_loss:
                early_stopping_counter = early_stopping_counter + 1
            
            if early_stopping_counter > early_stopping_patient:
                save_model(model, tokenizer)
                #stop_training = True

            
            model.train()

def model_validate(loader, device, model):
    val_loss = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype = torch.long)
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)

            val_loss.append(
            model(
                input_ids=ids,
                attention_mask=mask,
                labels=y).loss.item()
            )
    return np.mean(val_loss)

def save_model(model, tokenizer, step=0):
    
    print(f"[Saving Model]...\n")

    # Saving the model after training
    path = f'./tmp/{step}'
    create_dir(path)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    
     # save model for current epoch
    torch.save(model, f'{path}/mt5-base-ft.pth')

    # Upload to bucket
    upload_from_directory(path, BUCKET_NAME, JOB_NAME)
    print(f'Model save in gs://{BUCKET_NAME}/{JOB_NAME}/tmp/{step}')

def validate(epoch, tokenizer, model, device, loader):
  """
  Function to evaluate model for predictions. Generates a list of prediction and expected values. 
  """
  model.eval()
  predictions = []
  actuals = []
  with torch.no_grad():
      for _, data in enumerate(loader, 0):
          y = data['target_ids'].to(device, dtype = torch.long)
          ids = data['source_ids'].to(device, dtype = torch.long)
          mask = data['source_mask'].to(device, dtype = torch.long)

          generated_ids = model.generate(
              input_ids = ids,
              attention_mask = mask, 
              max_length=150, 
              num_beams=2,
              repetition_penalty=2.5, 
              length_penalty=1.0, 
              early_stopping=True
              )
          preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
          target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]
          if _ % 10==0:
              print(f'Completed {_}')

          predictions.extend(preds)
          actuals.extend(target)
  return predictions, actuals