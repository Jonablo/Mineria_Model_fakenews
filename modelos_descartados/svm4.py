import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_curve, f1_score
)
import joblib

# 1) Preparar stopwords
nltk.download("stopwords")
spanish_stopwords = stopwords.words("spanish")

# 2) Cargar y etiquetar
df_train = pd.read_csv("datos/train.csv")[["text","label"]]
df_fakes = pd.read_csv("datos/onlyfakes1000.csv")[["text"]]
df_true  = pd.read_csv("datos/onlytrue1000.csv")[["text"]]
df_fakes["label"] = 1
df_true ["label"] = 0

# 3) Concatenar y barajar
df_all = pd.concat([df_train, df_fakes, df_true], ignore_index=True)
df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)

# 4) Train/test split estratificado
X_train, X_test, y_train, y_test = train_test_split(
    df_all["text"], df_all["label"],
    test_size=0.2, random_state=42, stratify=df_all["label"]
)

# 5) Vectorizar
vectorizer = TfidfVectorizer(
    stop_words=spanish_stopwords,
    max_features=5_000,
    ngram_range=(1,2)
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf  = vectorizer.transform(X_test)

# 6) GridSearch para C de SVM con class_weight='balanced'
param_grid = {"C":[0.1,1,10,100]}
base_svm = SVC(
    kernel="linear",
    class_weight="balanced",
    probability=True,
    random_state=42
)
grid = GridSearchCV(
    base_svm, param_grid, cv=3,
    scoring="f1", n_jobs=-1, verbose=1
)
grid.fit(X_train_tfidf, y_train)
print("Mejor C:", grid.best_params_)

# 7) Calibración de probabilidades
svm_cal = CalibratedClassifierCV(
    grid.best_estimator_,   # se pasa el SVM mejor ajustado
    cv="prefit",
    method="isotonic"
)
svm_cal.fit(X_train_tfidf, y_train)

# 8) Encontrar umbral óptimo en validación (PR-curve)
probs_val = svm_cal.predict_proba(X_test_tfidf)[:,1]
prec, rec, th = precision_recall_curve(y_test, probs_val)
f1_scores = 2*prec*rec/(prec+rec+1e-9)
best_ix = np.argmax(f1_scores)
best_thr = th[best_ix]
print(f"Umbral óptimo para F1 en validación: {best_thr:.3f} (F1={f1_scores[best_ix]:.3f})")

# 9) Evaluación final con ese umbral
y_pred_thr = (probs_val >= best_thr).astype(int)
print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred_thr))
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred_thr))

# 10) Guardar modelo y vectorizador
joblib.dump(svm_cal, "svm_balanced_calibrated.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
print("Modelos guardados.")