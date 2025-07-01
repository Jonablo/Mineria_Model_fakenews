import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

# Carga datos combinados (ajusta la ruta)
df_train = pd.read_csv("datos/train.csv")
df_fakes = pd.read_csv("datos/onlyfakes1000.csv")
df_true = pd.read_csv("datos/onlytrue1000.csv")

df_fakes['label'] = 1
df_true['label'] = 0

df_fakes = df_fakes[['text', 'label']]
df_true = df_true[['text', 'label']]
df_train = df_train[['text', 'label']]

df_all = pd.concat([df_train, df_fakes, df_true], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

# Dividir train/test
train_df, test_df = train_test_split(df_all, test_size=0.2, stratify=df_all['label'], random_state=42)

# Convertir a datasets HuggingFace
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Tokenizador BETO
tokenizer = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True, max_length=512)

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset = train_dataset.remove_columns(["text", "__index_level_0__"])
test_dataset = test_dataset.remove_columns(["text", "__index_level_0__"])

train_dataset.set_format("torch")
test_dataset.set_format("torch")

# Métricas de evaluación
def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average='binary')
    acc = accuracy_score(p.label_ids, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# Modelo BETO para clasificación binaria
model = AutoModelForSequenceClassification.from_pretrained(
    "dccuchile/bert-base-spanish-wwm-cased", num_labels=2
)

training_args = TrainingArguments(
    output_dir="./results",
    save_steps=500,
    eval_steps=500,
    logging_steps=250,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.evaluate()