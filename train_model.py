from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)

import numpy as np


# -----------------------------
# LOAD DATASET
# -----------------------------

dataset = load_dataset(
    "json",
    data_files="agentcloud/data/train.jsonl"
)

dataset = dataset["train"].train_test_split(
    test_size=0.2,
    seed=42
)

train_dataset = dataset["train"]
test_dataset = dataset["test"]


# -----------------------------
# LABELS
# -----------------------------

labels = {
    "normal": 0,
    "overload": 1,
    "crash": 2,
    "intrusion": 3
}

id2label = {
    0: "normal",
    1: "overload",
    2: "crash",
    3: "intrusion"
}


# -----------------------------
# MODEL
# -----------------------------

model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(
    model_name
)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=4,
    id2label=id2label,
    label2id=labels
)


# -----------------------------
# PREPROCESSING
# -----------------------------

def preprocess(example):

    text = (
        f"{example.get('service', '')} "
        f"{example.get('level', '')} "
        f"{example.get('message', '')}"
    )

    tokenized = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128
    )

    tokenized["label"] = labels[
        example["label"]
    ]

    return tokenized


train_dataset = train_dataset.map(
    preprocess
)

test_dataset = test_dataset.map(
    preprocess
)


# -----------------------------
# REMOVE UNUSED COLUMNS
# -----------------------------

train_dataset = train_dataset.remove_columns(
    ["service", "message"]
)

test_dataset = test_dataset.remove_columns(
    ["service", "message"]
)
train_dataset.set_format("torch")
test_dataset.set_format("torch")


# -----------------------------
# TRAINING CONFIG
# -----------------------------

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    save_steps=100,
    logging_steps=10,
    learning_rate=2e-5,
    weight_decay=0.01
)


# -----------------------------
# METRICS
# -----------------------------

def compute_metrics(eval_pred):

    logits, labels_true = eval_pred

    predictions = np.argmax(
        logits,
        axis=-1
    )

    accuracy = accuracy_score(
        labels_true,
        predictions
    )

    precision = precision_score(
        labels_true,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        labels_true,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1_macro = f1_score(
        labels_true,
        predictions,
        average="macro",
        zero_division=0
    )

    f1_weighted = f1_score(
        labels_true,
        predictions,
        average="weighted",
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }


# -----------------------------
# TRAINER
# -----------------------------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)


# -----------------------------
# TRAIN
# -----------------------------

trainer.train()


# -----------------------------
# EVALUATION
# -----------------------------

predictions = trainer.predict(
    test_dataset
)

preds = np.argmax(
    predictions.predictions,
    axis=-1
)

true_labels = predictions.label_ids


# -----------------------------
# FINAL METRICS
# -----------------------------

accuracy = accuracy_score(
    true_labels,
    preds
)

precision = precision_score(
    true_labels,
    preds,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    true_labels,
    preds,
    average="weighted",
    zero_division=0
)

f1_macro = f1_score(
    true_labels,
    preds,
    average="macro",
    zero_division=0
)

f1_weighted = f1_score(
    true_labels,
    preds,
    average="weighted",
    zero_division=0
)


print("\n===== FINAL METRICS =====\n")

print(f"Accuracy        : {accuracy:.4f}")
print(f"Precision       : {precision:.4f}")
print(f"Recall          : {recall:.4f}")
print(f"F1 Macro Score  : {f1_macro:.4f}")
print(f"F1 Weighted     : {f1_weighted:.4f}")


# -----------------------------
# CLASSIFICATION REPORT
# -----------------------------

print("\n===== CLASSIFICATION REPORT =====\n")

print(
    classification_report(
        true_labels,
        preds,
        target_names=[
            "normal",
            "overload",
            "crash",
            "intrusion"
        ],
        zero_division=0
    )
)


# -----------------------------
# CONFUSION MATRIX
# -----------------------------

print("\n===== CONFUSION MATRIX =====\n")

print(
    confusion_matrix(
        true_labels,
        preds
    )
)


# -----------------------------
# SAVE MODEL
# -----------------------------

model.save_pretrained(
    "agentcloud/models/cloudsec-model"
)

tokenizer.save_pretrained(
    "agentcloud/models/cloudsec-model"
)

print("\nTraining complete.")