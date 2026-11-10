
import time
import pandas as pd
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    pipeline
)

from datasets import Dataset

MODEL_NAME = "distilbert-base-uncased"

df = pd.read_csv("data.csv")
df = df.dropna(subset=["text", "label"])
df["text"] = df["text"].astype(str).str.strip()
df["label"] = df["label"].astype(str).str.strip()
df = df[df["text"] != ""]

labels = sorted(df["label"].unique())
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}

df["label_id"] = df["label"].map(label2id)

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["label_id"]
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["label_id"]
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

sentences = [
    "I love this product",
    "The service is very bad"
]

for sentence in sentences:
    encoded = tokenizer(sentence)
    print("\nSentence:", sentence)
    print("Tokens:", tokenizer.convert_ids_to_tokens(encoded["input_ids"]))
    print("Input IDs:", encoded["input_ids"])
    print("Attention Mask:", encoded["attention_mask"])

sentiment_pipe = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

ner_pipe = pipeline(
    "ner",
    model="dslim/bert-base-NER",
    aggregation_strategy="simple"
)

sentiment_sentences = [
    "I love this product",
    "The service is excellent",
    "This is a terrible experience"
]

ner_sentences = [
    "Sundar works at Google in India",
    "Elon Musk founded Tesla",
    "Ravi lives in Chennai"
]

print("\nSENTIMENT ANALYSIS")

for sentence in sentiment_sentences:
    print(sentence)
    print(sentiment_pipe(sentence))

print("\nNAMED ENTITY RECOGNITION")

for sentence in ner_sentences:
    print(sentence)
    print(ner_pipe(sentence))

X_train = train_df["text"].tolist()
y_train = train_df["label_id"].tolist()

X_val = val_df["text"].tolist()
y_val = val_df["label_id"].tolist()

X_test = test_df["text"].tolist()
y_test = test_df["label_id"].tolist()

baseline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000)),
    ("model", LogisticRegression(max_iter=1000))
])

start = time.time()
baseline.fit(X_train, y_train)
baseline_train_time = time.time() - start

start = time.time()
baseline_predictions = baseline.predict(X_test)
baseline_inference_time = (time.time() - start) / len(X_test)

baseline_accuracy = accuracy_score(y_test, baseline_predictions)
baseline_f1 = f1_score(y_test, baseline_predictions, average="macro")

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=128
    )

def create_dataset(data):
    dataset = Dataset.from_pandas(
        data[["text", "label_id"]].rename(
            columns={"label_id": "labels"}
        ),
        preserve_index=False
    )
    dataset = dataset.map(tokenize_function, batched=True)
    dataset = dataset.remove_columns(["text"])
    return dataset

train_dataset = create_dataset(train_df)
val_dataset = create_dataset(val_df)
test_dataset = create_dataset(test_df)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)

def compute_metrics(eval_pred):
    predictions, actual = eval_pred
    predictions = np.argmax(predictions, axis=1)

    return {
        "accuracy": accuracy_score(actual, predictions),
        "f1": f1_score(
            actual,
            predictions,
            average="macro"
        )
    }

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="none",
    fp16=torch.cuda.is_available()
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return {
        "f1": f1_score(labels, predictions, average="weighted"),
        "accuracy": accuracy_score(labels, predictions)
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

start = time.time()
trainer.train()
training_time = time.time() - start

evaluation = trainer.evaluate(test_dataset)

start = time.time()
predictions = trainer.predict(test_dataset)
inference_time = (time.time() - start) / len(test_dataset)

transformer_predictions = np.argmax(
    predictions.predictions,
    axis=1
)

transformer_accuracy = accuracy_score(
    y_test,
    transformer_predictions
)

transformer_f1 = f1_score(
    y_test,
    transformer_predictions,
    average="macro"
)

trainer.save_model("./saved_model")
tokenizer.save_pretrained("./saved_model")

comparison = pd.DataFrame([
    {
        "Model": "TF-IDF",
        "Accuracy": baseline_accuracy,
        "Macro F1": baseline_f1,
        "Training Time": baseline_train_time,
        "Inference Time": baseline_inference_time
    },
    {
        "Model": "DistilBERT",
        "Accuracy": transformer_accuracy,
        "Macro F1": transformer_f1,
        "Training Time": training_time,
        "Inference Time": inference_time
    }
])

print("\nMODEL COMPARISON")
print(comparison.to_string(index=False))

comparison.to_csv("model_comparison.csv", index=False)



