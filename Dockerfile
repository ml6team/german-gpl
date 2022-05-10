# dockerfile for the training job in vertex ai

FROM europe-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-10:latest

WORKDIR /root

COPY requirements.txt /root
RUN pip install -r requirements.txt --no-cache

# set working directory
COPY ./src /app
WORKDIR /app

# Install required packages
RUN pip install google-cloud-storage transformers datasets tqdm cloudml-hypertune
RUN pip install wandb
RUN wandb login $WANDB_KEY

# Set up the entry point to invoke the trainer.
ENTRYPOINT ["python", "-m", "task"]