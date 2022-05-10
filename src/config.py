# T5 Training parameters
model_params = {
    "MODEL": "google/mt5-small",  # model_type: t5-base/t5-large
    "TRAIN_BATCH_SIZE": 8,  # training batch size
    "VALID_BATCH_SIZE": 8,  # validation batch size
    "TRAIN_EPOCHS": 1,  # number of training epochs
    "VAL_EPOCHS": 1,  # number of validation epochs
    "LEARNING_RATE": 1e-4, #1e-4,  # learning rate # try 0.001
    "MAX_SOURCE_TEXT_LENGTH": 512,  # max length of source text
    "MAX_TARGET_TEXT_LENGTH": 64,  # max length of target text
    "SEED": 42,  # set seed for reproducibility
}

dataset_params = {
    'DATASET_SIZE': 500000,
    'QUERY_COLUMN_NAME': 'query',
    'POSITIVE_COLUMN_NAME':'positive',
    'NEGATIVE_COLUMN_NAME': 'negative',
    'LANGUAGE': 'german'
}

# Defining the parameters for creation of dataloaders
train_params = {
    "batch_size": model_params["TRAIN_BATCH_SIZE"],
    "shuffle": True,
    "num_workers": 0,
    "drop_last": True
}

val_params = {
    "batch_size": model_params["VALID_BATCH_SIZE"],
    "shuffle": False,
    "num_workers": 0,
    "drop_last": True
}

# CrossEncoder Training parameters
crossencoder_params = {
    "model": "distilbert-base-multilingual-cased",
    "output_dir": "./out",
    "train_batch_size": 8,
    "num_epochs": 1,
    "warm_up_steps": 0,
}