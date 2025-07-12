import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_curve, f1_score
)
from sklearn.utils import resample
from scipy.sparse import hstack
from typing import Optional
import joblib
import openai
import time
from scipy.sparse import csr_matrix

# Carga las variables de .env al entorno
load_dotenv()

# ——————— 1) Config y stopwords ———————
nltk.download("stopwords", quiet=True)
SPANISH_STOPWORDS = stopwords.words("spanish")

# ——————— 2) LLM: extracción de hechos ———————
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise RuntimeError("Define la variable OPENAI_API_KEY")

def extract_facts(text: str) -> str:
    """Extrae 3 hechos breves con GPT y devuelve un string concatenado."""
    prompt = f"""Eres un extractor de hechos clave. De la noticia en español, extrae 3 frases breves, cada una con un solo hecho. Devuélvelas separadas por ‘;’."""
    resp = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user","content":prompt + "\n\n" + text}],
        temperature=0.2,
        max_tokens=150
    )
    # Extraer contenido seguro
    contenido = resp.choices[0].message.content or ""
    # Ahora nunca es None
    return contenido.replace("\n"," ").strip()

# ——————— 3) Spark: cargar Parquet etiquetado 0 ———————
PARQUET_PATH = "/workspaces/Mineria_Model_fakenews/data/noticias_rss_parquet"
WAREHOUSE_DIR = "/workspaces/Mineria_Model_fakenews/data/spark-warehouse"

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

# ——————— 4) Cargar CSVs originales ———————
def load_csv_data():
    base = "/workspaces/Mineria_Model_fakenews/datos"
    df_train = pd.read_csv(f"{base}/train.csv")[["text","label"]]
    df_fakes = pd.read_csv(f"{base}/onlyfakes1000.csv")[["text"]]
    df_true  = pd.read_csv(f"{base}/onlytrue1000.csv")[["text"]]
    df_fakes["label"] = 1
    df_true["label"]  = 0
    return df_train, df_fakes, df_true

# ——————— 5) Construir y limpiar dataset ———————
def build_dataset():
    df_spark = load_spark_data()
    df_train, df_fakes, df_true = load_csv_data()
    df_all = pd.concat([df_spark, df_train, df_fakes, df_true], ignore_index=True)
    df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)
    df_all = df_all.dropna(subset=["text","label"])
    df_all = df_all[df_all["text"].str.strip().str.len() > 10]
    return df_all

# ——————— 6) Augmentación y oversampling ———————
def augment_and_oversample(df: pd.DataFrame, llm_limit: Optional[int] = None) -> pd.DataFrame:
    # 6a) Extraer hechos con LLM para todos
    cache = {}
    augmented = []
    for i, txt in enumerate(df["text"]):
        if llm_limit is None or i < llm_limit:
            if txt not in cache:
                cache[txt] = extract_facts(txt)
                time.sleep(1)  # para no saturar la API
            augmented.append(txt + " " + cache[txt])
        else:
            augmented.append(txt)
    df["aug_text"] = augmented

    # 6b) Oversample clase 1 (fakes) para igualar a la clase 0
    df_0 = df[df.label == 0]
    df_1 = df[df.label == 1]
    if len(df_1) < len(df_0):
        df_1 = resample(df_1,
                        replace=True,
                        n_samples=len(df_0),
                        random_state=42)
    # Ensure both are DataFrames
    df_0 = pd.DataFrame(df_0)
    df_1 = pd.DataFrame(df_1)
    return pd.concat([df_0, df_1], ignore_index=True).sample(frac=1, random_state=42)

# ——————— 7) Features adicionales ———————
def compute_features(texts: pd.Series) -> np.ndarray:
    lengths = texts.str.len().to_numpy().reshape(-1,1)
    uppercase_ratio = texts.str.count(r"[A-Z]").to_numpy().reshape(-1,1) / (lengths + 1)
    return np.hstack([lengths, uppercase_ratio])

# ——————— 8) Pipeline principal ———————
def main():
    # a) Dataset
    df = build_dataset()
    df = augment_and_oversample(df, llm_limit=None)  # llm_limit=None => todas
    print(f"Total ejemplos tras oversample: {len(df)}")

    X_text = df["aug_text"]
    y = df["label"]

    # b) Split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_text, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # c) TF-IDF + features
    vec = TfidfVectorizer(stop_words=SPANISH_STOPWORDS,
                          max_features=5000, ngram_range=(1,2))
    X_tr_tfidf = vec.fit_transform(X_tr)
    X_te_tfidf = vec.transform(X_te)

    feat_tr = compute_features(X_tr)
    feat_te = compute_features(X_te)

    # Combinar sparse TF-IDF + numéricas


    X_tr_full = csr_matrix(hstack([X_tr_tfidf, csr_matrix(feat_tr)]))
    X_te_full = csr_matrix(hstack([X_te_tfidf, csr_matrix(feat_te)]))

    # d) GridSearch + StratifiedKFold
    param_grid = {"C":[0.1,1,10,100]}
    base = SVC(kernel="linear", class_weight={0:1,1:2},
               probability=True, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(base, param_grid, cv=cv, scoring="f1",
                        n_jobs=-1, verbose=1)
    grid.fit(X_tr_full, y_tr)
    print("Mejor C:", grid.best_params_)

    # e) Calibración de probas
    cal = CalibratedClassifierCV(grid.best_estimator_,
                                 cv="prefit", method="isotonic")
    cal.fit(X_tr_full, y_tr)

    # f) Umbral óptimo
    p_val = cal.predict_proba(X_te_full)[:,1]
    prec, rec, th = precision_recall_curve(y_te, p_val)
    f1s = 2*prec*rec/(prec+rec+1e-9)
    ix = np.argmax(f1s)
    thr = th[ix]
    print(f"Umbral (max F1): {thr:.3f}, F1={f1s[ix]:.3f}")

    # g) Evaluación final
    y_pred = (p_val >= thr).astype(int)
    print("\nMatriz de confusión:")
    print(confusion_matrix(y_te, y_pred))
    print("\nReporte de clasificación:")
    print(classification_report(y_te, y_pred))

    # h) Guardar
    joblib.dump(cal, "svm_enhanced_calibrated.pkl")
    joblib.dump(vec, "tfidf_enhanced_vectorizer.pkl")
    joblib.dump(thr, "best_thr.pkl")
    print("Modelos y umbral guardados.")

if __name__ == "__main__":
    main()