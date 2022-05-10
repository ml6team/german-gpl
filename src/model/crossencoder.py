from torch.utils.data import DataLoader
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
from sentence_transformers import InputExample
from data.dataset import create_dataframe, download_mmarco
from config import *
from utils import create_dir, upload_from_directory


# We train the network with as a binary label task
# Given [query, passage] is the label 0 = irrelevant or 1 = relevant?
# We use a positive-to-negative ratio: For 1 positive sample (label 1) we include 4 negative samples (label 0)
# in our training setup. For the negative samples, we use the triplets provided by MS Marco that
# specify (query, positive sample, negative sample).

def get_data():
    """
    Retrieve dataset of bigquery and create InputExamples of <query, positive, 1> and <query, negative, 0>.
    Return train and test samples. 
    """
    data = download_mmarco()
    df_train, df_val = create_dataframe(data, 
        dataset_params['QUERY_COLUMN_NAME'], 
        dataset_params['POSITIVE_COLUMN_NAME'],
        dataset_params['NEGATIVE_COLUMN_NAME'],
        dataset_params['DATASET_SIZE']
        )
    train_samples = create_samples_of_df(df_train)
    eval_samples = create_samples_of_df(df_val)
    return train_samples, eval_samples

def create_samples_of_df(df):
    samples = []
    for index, row in df.iterrows():
        samples.append(InputExample(texts=[row['query'], row['positive']], label=1))
        samples.append(InputExample(texts=[row['query'], row['negative']], label=0))
    return samples

def save_model(model):
    create_dir(crossencoder_params['output_dir'])
    model.save(crossencoder_params['output_dir'])

def run():
    model = CrossEncoder(crossencoder_params['model'], num_labels=1)

    # dataloader
    train_samples, eval_samples = get_data()
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=crossencoder_params['train_batch_size'])

    # Train the model
    create_dir(crossencoder_params['output_dir'])
    model.fit(train_dataloader=train_dataloader,
            epochs=crossencoder_params['num_epochs'],
            warmup_steps=crossencoder_params['warm_up_steps'])

    # save model
    print(f'Training done. Save model ... ')
    save_model(model)

    ##### Load model and eval on test set
    print('Start model evaluation')
    model = CrossEncoder(crossencoder_params['output_dir'])

    evaluator = CECorrelationEvaluator.from_input_examples(eval_samples, name='sts-test')
    print(evaluator(model))

