from datasets import load_dataset, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd

# 1. Carga de datos
df = pd.read_csv("datos/train.csv")

# Convierte pandas a Dataset Hugging Face
dataset = Dataset.from_pandas(df)

# Split manual train/test (80/20)
dataset = dataset.train_test_split(test_size=0.2, seed=42)

# 2. Tokenizador BETO
tokenizer = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True, max_length=512)

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# 3. Modelo BETO para clasificación binaria
model = AutoModelForSequenceClassification.from_pretrained("dccuchile/bert-base-spanish-wwm-cased", num_labels=2)

# 4. Métricas para evaluación
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}

# 5. Configuración de entrenamiento
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1"
)

# 6. Entrenador
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics
)

# 7. Entrenamiento
trainer.train()

# 8. Evaluación final
metrics = trainer.evaluate()
print(metrics)
