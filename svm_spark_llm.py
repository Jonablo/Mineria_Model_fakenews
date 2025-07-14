import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    f1_score
)
from sklearn.utils import resample
from scipy.sparse import hstack, csr_matrix
import joblib
import openai
import time

# Carga .env
load_dotenv()

# 1) Configuración de stopwords y OpenAI
nltk.download("stopwords", quiet=True)
SPANISH_STOPWORDS = stopwords.words("spanish")
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise RuntimeError("Define OPENAI_API_KEY en tu .env")

# 2) Función LLM para extraer hechos
def extract_facts(text: str) -> str:
    prompt = (
        "Eres un extractor de hechos clave. De la noticia en español, "
        "extrae 3 frases breves, cada una con un solo hecho. "
        "Devuélvelas separadas por ‘;’. \n\n" + text
    )
    resp = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=150
    )
    return (resp.choices[0].message.content or "").replace("\n", " ").strip()

# 3) Leer Parquet desde Spark (etiqueta=0)
PARQUET_PATH = "data/noticias_rss_parquet"
WAREHOUSE_DIR = "data/spark-warehouse"

def create_spark_session():
    return SparkSession.builder \
        .appName("LoadRSSParquet") \
        .config("spark.sql.warehouse.dir", WAREHOUSE_DIR) \
        .getOrCreate()

def load_spark_data() -> pd.DataFrame:
    spark = create_spark_session()
    df = (
        spark.read.parquet(PARQUET_PATH)
             .withColumn("label", lit(0))
             .select("text", "label")
    )
    pdf = df.toPandas()
    spark.stop()
    return pdf

# 4) Cargar CSVs
def load_csv_data():
    base = "datos"
    df_train = pd.read_csv(f"{base}/train.csv")[["text","label"]]
    df_fakes = pd.read_csv(f"{base}/onlyfakes1000.csv")[["text"]]; df_fakes["label"]=1
    df_true  = pd.read_csv(f"{base}/onlytrue1000.csv")[["text"]];  df_true["label"]=0
    return df_train, df_fakes, df_true

# 5) Construir y limpiar dataset
def build_dataset() -> pd.DataFrame:
    df_spark = load_spark_data()
    df_train, df_fakes, df_true = load_csv_data()
    df_all = pd.concat([df_spark, df_train, df_fakes, df_true], ignore_index=True)
    df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)
    df_all = df_all.dropna(subset=["text","label"])
    df_all = df_all[df_all["text"].str.strip().str.len() > 10]
    return df_all

# 6) Augmentación + oversample
def augment_and_oversample(df: pd.DataFrame, llm_limit: int = 300) -> pd.DataFrame:
    cache, augmented = {}, []
    for i, txt in enumerate(df["text"]):
        if i < llm_limit:
            if txt not in cache:
                cache[txt] = extract_facts(txt)
                time.sleep(1)
            augmented.append(f"{txt} {cache[txt]}")
        else:
            augmented.append(txt)
    df["aug_text"] = augmented

    df0 = df[df.label == 0]
    df1 = df[df.label == 1]
    if len(df1) < len(df0):
        df1 = resample(df1, replace=True, n_samples=len(df0), random_state=42)
    return pd.concat([df0, df1], ignore_index=True).sample(frac=1, random_state=42)

# 7) Características numéricas adicionales
def compute_features(texts: pd.Series) -> np.ndarray:
    lengths = texts.str.len().to_numpy().reshape(-1,1)
    upper_ratio = texts.str.count(r"[A-Z]").to_numpy().reshape(-1,1) / (lengths + 1)
    return np.hstack([lengths, upper_ratio])

# 8) Pipeline principal
def main():
    df = build_dataset()
    df = augment_and_oversample(df, llm_limit=300)
    print(f"Total tras oversample: {len(df)}")

    X, y = df["aug_text"], df["label"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    vec = TfidfVectorizer(
        stop_words=SPANISH_STOPWORDS,
        max_features=5000,
        ngram_range=(1,2)
    )
    X_tr_tfidf = vec.fit_transform(X_tr)
    X_te_tfidf = vec.transform(X_te)

    feat_tr = compute_features(X_tr)
    feat_te = compute_features(X_te)

    X_tr_full = csr_matrix(hstack([X_tr_tfidf, csr_matrix(feat_tr)]))
    X_te_full = csr_matrix(hstack([X_te_tfidf, csr_matrix(feat_te)]))

    # 9) Entrenar + calibrar directamente
    svc = LinearSVC(
        C=1,
        class_weight={0:1, 1:2},
        max_iter=10_000,
        random_state=42
    )
    cal = CalibratedClassifierCV(
        estimator=svc,
        cv=3,
        method="isotonic"
    )
    cal.fit(X_tr_full, y_tr)

    probs = cal.predict_proba(X_te_full)[:,1]
    prec, rec, th = precision_recall_curve(y_te, probs)
    f1s = 2 * prec * rec / (prec + rec + 1e-9)
    best_ix = np.argmax(f1s)
    best_thr = th[best_ix]
    print(f"Umbral óptimo: {best_thr:.3f}, F1={f1s[best_ix]:.3f}")

    y_pred = (probs >= best_thr).astype(int)
    print("\nMatriz de confusión:")
    print(confusion_matrix(y_te, y_pred))
    print("\nReporte final:")
    print(classification_report(y_te, y_pred))

    joblib.dump(cal, "svm_enhanced_calibrated.pkl")
    joblib.dump(vec, "tfidf_enhanced_vectorizer.pkl")
    joblib.dump(best_thr, "best_thr.pkl")
    print("Modelos y umbral guardados.")

if __name__ == "__main__":
    main()