import os
import datasets
import pandas as pd
from simpletransformers.classification import ClassificationModel


os.environ["TOKENIZERS_PARALLELISM"] = "false"

dataset_dir = 'data/emotion'

train_df = pd.read_csv(f'{dataset_dir}/gen_small.csv')
eval_df = pd.read_csv(f'{dataset_dir}/dev.csv')

# Create a ClassificationModel
model = ClassificationModel(
    "roberta",
    "roberta-base",
    num_labels=6,
    args={
        "reprocess_input_data": True,
        "overwrite_output_dir": True,
        "train_batch_size": 16,
        "eval_batch_size": 16,
        "num_train_epochs": 10,
        "n_gpu": 1,
        # "wandb_project": "CS260D",
        "evaluate_during_training" : True,
        "evaluate_during_training_steps" : 2000,
        "use_multiprocessing": False,
        "use_multiprocessing_for_evaluation": False,
    },
)

# Train the model
model.train_model(
    train_df=train_df,
    eval_df=eval_df,
    args={
        "gradient_record_interval": 1,
    },
)
