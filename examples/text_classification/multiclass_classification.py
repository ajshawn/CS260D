import pandas as pd

from simpletransformers.classification import ClassificationModel

# Train and Evaluation data needs to be in a Pandas Dataframe containing at least two columns. If the Dataframe has a header, it should contain a 'text' and a 'labels' column. If no header is present, the Dataframe should contain at least two columns, with the first column is the text with type str, and the second column in the label with type int.
train_data = [
    ["Example sentence belonging to class 1", 1],
    ["Example sentence belonging to class 0", 0],
    ["Example eval senntence belonging to class 2", 2],
    ["Example eval belonging to class 2", 2],
]
train_df = pd.DataFrame(train_data)

eval_data = [
    ["Example eval sentence belonging to class 1", 1],
    ["Example eval sentence belonging to class 0", 0],
    ["Example eval senntence belonging to class 2", 2],
    ["Example eval belonging to class 2", 2],
]
eval_df = pd.DataFrame(eval_data)

# Create a ClassificationModel
model = ClassificationModel(
    "bert",
    "bert-base-cased",
    num_labels=3,
    args={
        "reprocess_input_data": True,
        "overwrite_output_dir": True,
        "train_batch_size": 2,
        "eval_batch_size": 2,
        "num_train_epochs": 10,
        "n_gpu": 1,
    },
)

# Train the model
model.train_model(
    train_df,
    args={
        "gradient_record_interval": 2,
    },
)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(eval_df)

predictions, raw_outputs = model.predict(["Some arbitary sentence"])
