import os
import time
import joblib
import numpy as np
import pandas as pd
import nltk
from typing import Optional
from nltk.corpus import stopwords
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, f1_score
from sklearn.utils import resample
from scipy.sparse import hstack, csr_matrix
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit

# ————— 1) STOPWORDS —————
nltk.download("stopwords", quiet=True)
SPANISH_STOPWORDS = stopwords.words("spanish")

# ————— 2) SPARK: Carga Parquet como “verdaderas” —————
PARQUET_PATH   = "data/noticias_rss_parquet"
WAREHOUSE_DIR  = "data/spark-warehouse"

def create_spark_session():
    return SparkSession.builder \
        .appName("LoadRSSParquet") \
        .config("spark.sql.warehouse.dir", WAREHOUSE_DIR) \
        .getOrCreate()

def load_spark_data() -> pd.DataFrame:
    spark = create_spark_session()
    df = spark.read.parquet(PARQUET_PATH) \
              .withColumn("label", lit(0)) \
              .select("text", "label")
    pdf = df.toPandas()
    spark.stop()
    return pdf

# ————— 3) CSVs de train —————
def load_csv_data():
    df_train = pd.read_csv("datos/train.csv")[["text","label"]]
    df_fakes = pd.read_csv("datos/onlyfakes1000.csv")[["text"]]
    df_true  = pd.read_csv("datos/onlytrue1000.csv")[["text"]]
    df_fakes["label"] = 1
    df_true ["label"] = 0
    return df_train, df_fakes, df_true

# ————— 4) Construye dataset limpio —————
def build_dataset() -> pd.DataFrame:
    df_spark, = [load_spark_data()]               # noticias “reales”
    df_train, df_fakes, df_true = load_csv_data() # originales
    df_all = pd.concat([df_spark, df_train, df_fakes, df_true], ignore_index=True)
    df_all = df_all.dropna(subset=["text","label"])
    df_all = df_all[df_all["text"].str.strip().str.len() > 10]
    return df_all.sample(frac=1, random_state=42).reset_index(drop=True)

# ————— 5) Oversample para balancear —————
def oversample(df: pd.DataFrame) -> pd.DataFrame:
    df0 = df[df.label==0]
    df1 = df[df.label==1]
    if len(df1) < len(df0):
        df1 = resample(df1, replace=True, n_samples=len(df0), random_state=42)
    return pd.concat([df0, df1], ignore_index=True).sample(frac=1, random_state=42)

# ————— 6) Característica auxiliar —————
def compute_features(texts: pd.Series) -> np.ndarray:
    lengths = texts.str.len().to_numpy().reshape(-1,1)
    uppers  = texts.str.count(r"[A-Z]").to_numpy().reshape(-1,1)
    return np.hstack([lengths, uppers / (lengths+1)])

# ————— 7) Pipeline principal —————
def main():
    # a) Dataset + oversample
    df = build_dataset()
    df = oversample(df)
    print(f"Total ejemplos tras oversample: {len(df)}")

    # b) Train/test split
    X = df["text"]
    y = df["label"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # c) TF-IDF con n-gramas hasta trigrama y más features
    vectorizer = TfidfVectorizer(
        stop_words=SPANISH_STOPWORDS,
        max_features=10000,
        ngram_range=(1,3)
    )
    X_tr_tfidf = vectorizer.fit_transform(X_tr)
    X_te_tfidf = vectorizer.transform(X_te)

    feat_tr = compute_features(X_tr)
    feat_te = compute_features(X_te)

    X_tr_full = csr_matrix(hstack([X_tr_tfidf, csr_matrix(feat_tr)]))
    X_te_full = csr_matrix(hstack([X_te_tfidf, csr_matrix(feat_te)]))

    # d) GridSearch sobre LinearSVC alto max_iter
    base = LinearSVC(
        class_weight={0:1, 1:2},
        max_iter=50_000,
        dual=False,
        random_state=42
    )
    param_grid = {"C":[0.1,1,10,100]}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        estimator=base,
        param_grid=param_grid,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_tr_full, y_tr)
    print("Mejor C:", grid.best_params_)

    # e) Calibración (sigmoid más estable)
    svc = grid.best_estimator_
    cal = CalibratedClassifierCV(svc, cv=cv, method="sigmoid")
    cal.fit(X_tr_full, y_tr)

    # f) Umbral óptimo (max F1)
    probs = cal.predict_proba(X_te_full)[:,1]
    prec, rec, th = precision_recall_curve(y_te, probs)
    f1s = 2*prec*rec/(prec+rec+1e-9)
    ix = np.argmax(f1s)
    best_thr = th[ix]
    print(f"Umbral óptimo: {best_thr:.3f}, F1={f1s[ix]:.3f}")

    # g) Evaluación final
    y_pred = (probs >= best_thr).astype(int)
    print("\nMatriz de confusión:")
    print(confusion_matrix(y_te, y_pred))
    print("\nReporte final:")
    print(classification_report(y_te, y_pred, digits=4))

    # h) Guardar artefactos
    joblib.dump(cal,         "svm_enhanced_calibrated.pkl")
    joblib.dump(vectorizer,  "tfidf_enhanced_vectorizer.pkl")
    joblib.dump(best_thr,    "best_thr.pkl")
    print("Modelos y umbral guardados.")

if __name__ == "__main__":
    main()