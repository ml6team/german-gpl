# German GPL


## Install requirements

Run the following commands to install all needed requirements. 

```
pip install -r requirements.txt
```

During the training we will log information to [wandb](wandb.ai). Therefore, you have to export the API key to your local environment. 

```
export WANDB_KEY=<your_key>
```

## Run Training

Start cross-encoder fine-tuning

```
python -m task train_cross_encoder
```

Start mT5 fine-tuning
```
python -m task train_t5
```
