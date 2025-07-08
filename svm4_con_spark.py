import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_curve, f1_score
)
import joblib

from pyspark.sql import SparkSession
from pyspark.sql.functions import lit

# 1) Descargar stopwords
nltk.download("stopwords")
spanish_stopwords = stopwords.words("spanish")

# 2) Spark: lee tu Parquet y etiqueta TODO con 0
PARQUET_PATH = "./data/noticias_rss_parquet"
def create_spark_session():
    return SparkSession.builder \
        .appName("LoadRSSParquet") \
        .config("spark.sql.warehouse.dir", "./data/spark-warehouse") \
        .getOrCreate()

def load_spark_data(parquet_path: str) -> pd.DataFrame:
    spark = create_spark_session()
    df_spark = spark.read.parquet(parquet_path)
    # todas las noticias como "true" (0)
    df_labeled = df_spark.withColumn("label", lit(0))
    pdf = df_labeled.select("text", "label").toPandas()
    spark.stop()
    return pdf

# 3) Carga tus CSVs de entrenamiento
def load_csv_data():
    # tu CSV de train original, con su columna label
    df_train = pd.read_csv("datos/train.csv")[["text","label"]]
    # solo "fakes" y solo "true"
    df_fakes = pd.read_csv("datos/onlyfakes1000.csv")[["text"]]
    df_true  = pd.read_csv("datos/onlytrue1000.csv")[["text"]]
    df_fakes["label"] = 1
    df_true ["label"] = 0
    return df_train, df_fakes, df_true

# 4) Arma el DataFrame final
def build_dataset():
    df_spark = load_spark_data(PARQUET_PATH)
    df_train, df_fakes, df_true = load_csv_data()
    # concatena TODO
    df_all = pd.concat([df_spark, df_train, df_fakes, df_true], ignore_index=True)
    # baraja
    df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)
    # filtra nulos y textos muy cortos
    df_all = df_all.dropna(subset=["text","label"])
    df_all = df_all[df_all["text"].str.strip().str.len() > 10]
    return df_all

def main():
    df_all = build_dataset()
    X = df_all["text"]
    y = df_all["label"]
    print(f"Total ejemplos: {len(df_all)}")

    # 5) Train/test split estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 6) TF-IDF
    vectorizer = TfidfVectorizer(
        stop_words=spanish_stopwords,
        max_features=5_000,
        ngram_range=(1,2)
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf  = vectorizer.transform(X_test)

    # 7) GridSearch SVM con class_weight='balanced'
    param_grid = {"C": [0.1, 1, 10, 100]}
    base_svm = SVC(
        kernel="linear",
        class_weight="balanced",
        probability=True,
        random_state=42
    )
    grid = GridSearchCV(
        base_svm,
        param_grid,
        cv=3,
        scoring="f1",
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train_tfidf, y_train)
    print("Mejor C en SVM:", grid.best_params_)

    # 8) Calibración de probabilidades
    svm_cal = CalibratedClassifierCV(
        grid.best_estimator_,
        cv="prefit",
        method="isotonic"
    )
    svm_cal.fit(X_train_tfidf, y_train)

    # 9) Busca umbral óptimo con Precision-Recall
    probs_val = svm_cal.predict_proba(X_test_tfidf)[:,1]
    prec, rec, th = precision_recall_curve(y_test, probs_val)
    f1_scores = 2 * prec * rec / (prec + rec + 1e-9)
    best_ix = np.argmax(f1_scores)
    best_thr = th[best_ix]
    print(f"Umbral óptimo (F1): {best_thr:.3f}, F1={f1_scores[best_ix]:.3f}")

    # 10) Evaluación final
    y_pred_thr = (probs_val >= best_thr).astype(int)
    print("\nMatriz de confusión:")
    print(confusion_matrix(y_test, y_pred_thr))
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred_thr))

    # 11) Guardar modelo y vectorizador
    joblib.dump(svm_cal, "svm_balanced_calibrated.pkl")
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
    print("Modelos guardados a disco.")

if __name__ == "__main__":
    main()