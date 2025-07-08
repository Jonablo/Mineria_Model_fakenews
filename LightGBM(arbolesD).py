from scipy import sparse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from scipy.sparse import hstack
import nltk
from nltk.corpus import stopwords
import string
import lightgbm as lgb
import numpy as np

# Descargar stopwords si no están
nltk.download("stopwords")
spanish_stopwords = stopwords.words("spanish")

# Cargar dataset
df = pd.read_csv("datos/train.csv")

# Limpieza básica: eliminar duplicados y nulos
df = df.drop_duplicates(subset=["text"])
df = df.dropna(subset=["text", "label"])
df = df[df["text"].str.strip().str.len() > 5]

print(f"Total registros usados: {len(df)}")
print("Distribución clases:")
print(df["label"].value_counts())


def extract_basic_features(texts):
    features = pd.DataFrame()
    features["text_len"] = texts.apply(lambda x: len(x.split()))
    features["count_question"] = texts.apply(lambda x: x.count("?"))
    features["count_exclamation"] = texts.apply(lambda x: x.count("!"))
    features["count_punct"] = texts.apply(
        lambda x: sum(1 for c in x if c in string.punctuation)
    )
    return features


X_text = df["text"]
y = df["label"]
extra_features = extract_basic_features(X_text)

# Vectorizador TF-IDF
vectorizer = TfidfVectorizer(
    stop_words=spanish_stopwords, max_features=500, ngram_range=(1, 2)
)
X_tfidf = vectorizer.fit_transform(X_text)

extra_sparse = sparse.csr_matrix(extra_features.values)
X_full = hstack([X_tfidf, extra_sparse]).tocsr()

# Validación cruzada estratificada
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

acc_scores = []
f1_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, y), 1):

    X_train, X_val = X_full[train_idx], X_full[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = lgb.LGBMClassifier(
        n_estimators=100, max_depth=10, learning_rate=0.1, random_state=42, n_jobs=1
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    acc_scores.append(model.score(X_val, y_val))

print("Matriz de confusión:")
print(confusion_matrix(y_val, y_pred))

print("Reporte clasificación:")
print(classification_report(y_val, y_pred, zero_division=0))
print(f"\nAccuracy promedio CV: {np.mean(acc_scores):.4f}")
