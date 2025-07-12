import os
import pandas as pd
import numpy as np
import joblib
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader, Dataset

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

from transformers import AutoTokenizer, AutoModel

# ---------------------------
#    1) Configuración
# ---------------------------

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BETO_MODEL = "dccuchile/bert-base-spanish-wwm-cased"
BATCH_SIZE = 16

np.random.seed(SEED)
torch.manual_seed(SEED)


# ---------------------------
#   2) Carga de datos
# ---------------------------

# Ajusta rutas si hiciera falta:
df_train = pd.read_csv("datos/train.csv")
df_fakes = pd.read_csv("datos/onlyfakes1000.csv")
df_true  = pd.read_csv("datos/onlytrue1000.csv")

# Asignar etiquetas
df_fakes['label'] = 1
df_true['label']  = 0

# Seleccionar columnas
df_train = df_train[['text', 'label']]
df_fakes = df_fakes[['text', 'label']]
df_true  = df_true [['text', 'label']]

# Concatenar y barajar
df_all = pd.concat([df_train, df_fakes, df_true], ignore_index=True)
df_all = df_all.sample(frac=1, random_state=SEED).reset_index(drop=True)

print("Total textos:", len(df_all))
print(df_all.label.value_counts(), "\n")


# ---------------------------
#    3) Dataset PyTorch
# ---------------------------

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts.tolist()
        self.labels = labels.tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        txt = self.texts[idx]
        enc = self.tokenizer(
            txt,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        # devolver ids y máscara
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }


# ---------------------------
#   4) Carga modelo BETO
# ---------------------------

print("→ Cargando tokenizer y modelo BETO...")
tokenizer = AutoTokenizer.from_pretrained(BETO_MODEL)
model_beto = AutoModel.from_pretrained(BETO_MODEL)
model_beto.to(DEVICE)
model_beto.eval()


# ---------------------------
#  5) División train/test
# ---------------------------

train_df, test_df = train_test_split(
    df_all,
    test_size=0.2,
    stratify=df_all["label"],
    random_state=SEED
)

train_ds = TextDataset(train_df["text"], train_df["label"], tokenizer)
test_ds  = TextDataset(test_df ["text"], test_df ["label"], tokenizer)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)


# ---------------------------
#  6) Función para extraer embeddings CLS
# ---------------------------

@torch.no_grad()
def extract_cls_embeddings(dataloader):
    all_embs = []
    all_labels = []
    for batch in tqdm(dataloader, desc="Extrayendo CLS..."):
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["label"].numpy()

        outputs = model_beto(input_ids=input_ids, attention_mask=attention_mask)
        # outputs.last_hidden_state = (batch_size, seq_len, hidden_size)
        # tomamos la posición 0 = [CLS]
        cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embs.append(cls_emb)
        all_labels.append(labels)

    embeddings = np.vstack(all_embs)
    labels_np  = np.concatenate(all_labels)
    return embeddings, labels_np


# ---------------------------
#  7) Extraer embeddings
# ---------------------------

X_train_emb, y_train_np = extract_cls_embeddings(train_loader)
X_test_emb,  y_test_np  = extract_cls_embeddings(test_loader)

print("Shape train embeddings:", X_train_emb.shape)
print("Shape test  embeddings:", X_test_emb.shape)


# ---------------------------
#  8) Entrenar SVM (GridSearch)
# ---------------------------

print("\n→ Ajustando hiperparámetros de SVM...")
param_grid = {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}
svm = GridSearchCV(SVC(probability=True, random_state=SEED),
                   param_grid, cv=3, scoring="f1", n_jobs=-1, verbose=1)
svm.fit(X_train_emb, y_train_np)
print("Mejores params SVM:", svm.best_params_)

best_svm = svm.best_estimator_


# ---------------------------
#  9) Evaluación final
# ---------------------------

y_pred = best_svm.predict(X_test_emb)
y_proba = best_svm.predict_proba(X_test_emb)[:, 1]

print("\nMatriz de confusión:")
print(confusion_matrix(y_test_np, y_pred))
print("\nReporte clasificación:")
print(classification_report(y_test_np, y_pred))


# ---------------------------
# 10) Guardar modelo+tokenizer y SVM
# ---------------------------

os.makedirs("output", exist_ok=True)
print("\n→ Guardando artefactos en /output …")
model_beto.save_pretrained("output/beto_embedding_model")
tokenizer.save_pretrained("output/beto_embedding_tokenizer")
joblib.dump(best_svm,  "output/svm_beto.pkl")


print("\n¡Listo! ")