import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import SaeConfig, TrainConfig
from trainer import  SaeTrainer
from data import chunk_and_tokenize

dataset_name = "mbakler/test_dataset"

dataset = load_dataset(
    dataset_name,
    trust_remote_code=True,
    split = "batch_0_781"
)
print(len(dataset))

d_model=1024

cfg = TrainConfig(
    SaeConfig(), batch_size=16, embedding_task=True
)
# dead_feature_threshold questionable, how todo this???
trainer = SaeTrainer(cfg, dataset, None)
#
#trainer.fit()