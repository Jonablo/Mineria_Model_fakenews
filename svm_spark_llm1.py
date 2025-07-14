import os
import json
import time
import joblib
import openai
import pandas as pd
import numpy as np
import nltk
import concurrent.futures

from dotenv import load_dotenv
from nltk.corpus import stopwords
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

# ————— Setup —————
load_dotenv()
nltk.download("stopwords", quiet=True)
SPANISH_STOPWORDS = stopwords.words("spanish")

openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise RuntimeError("Define OPENAI_API_KEY en tu .env")

CACHE_FILE = "facts_cache.json"
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r") as f:
        FACTS_CACHE = json.load(f)
else:
        FACTS_CACHE = {}

# ————— 1) Leer Parquet con pandas como “verdaderas” —————
def load_parquet_data() -> pd.DataFrame:
    df = pd.read_parquet("data/noticias_rss_parquet")
    df["label"] = 0
    return df[["text", "label"]]

# ————— 2) CSVs de entrenamiento —————
def load_csv_data():
    base = "datos"
    df_train = pd.read_csv(f"{base}/train.csv")[["text","label"]]
    df_fakes = pd.read_csv(f"{base}/onlyfakes1000.csv")[["text"]]
    df_true  = pd.read_csv(f"{base}/onlytrue1000.csv")[["text"]]
    df_fakes["label"] = 1
    df_true ["label"] = 0
    return df_train, df_fakes, df_true

# ————— 3) Construir dataset limpio —————
def build_dataset() -> pd.DataFrame:
    df_parquet = load_parquet_data()
    df_train, df_fakes, df_true = load_csv_data()
    df_all = pd.concat([df_parquet, df_train, df_fakes, df_true], ignore_index=True)
    df_all = df_all.dropna(subset=["text","label"])
    df_all = df_all[df_all["text"].str.strip().str.len() > 10]
    return df_all.sample(frac=1, random_state=42).reset_index(drop=True)

# ————— 4) LLM extractor con cache —————
def extract_facts(text: str) -> str:
    if text in FACTS_CACHE:
        return FACTS_CACHE[text]
    prompt = (
        "Eres un extractor de hechos clave. De la noticia en español, "
        "extrae 3 frases breves, cada una con un solo hecho. "
        "Devuélvelas separadas por ‘;’. \n\n" + text
    )
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user","content":prompt}],
        temperature=0.2,
        max_tokens=150
    )
    facts = (resp.choices[0].message.content or "").replace("\n", " ").strip()
    FACTS_CACHE[text] = facts
    with open(CACHE_FILE, "w") as f:
        json.dump(FACTS_CACHE, f, ensure_ascii=False, indent=2)
    return facts

# ————— 5) Augmentación + oversample con paralelización —————
def augment_and_oversample(df: pd.DataFrame, llm_limit: int = 300) -> pd.DataFrame:
    texts = df["text"].tolist()
    to_enrich = texts[:llm_limit]
    unique = list(dict.fromkeys(to_enrich))
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(extract_facts, txt): txt for txt in unique}
        for future in concurrent.futures.as_completed(futures):
            txt = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error LLM en «{txt[:30]}…»: {e}")
    augmented = [
        f"{txt} {FACTS_CACHE.get(txt, '')}" if i < llm_limit else txt
        for i, txt in enumerate(texts)
    ]
    df["aug_text"] = augmented
    df0 = df[df.label == 0]
    df1 = df[df.label == 1]
    if len(df1) < len(df0):
        df1 = resample(df1, replace=True, n_samples=len(df0), random_state=42)
    df_bal = pd.concat([df0, df1], ignore_index=True)
    return df_bal.sample(frac=1, random_state=42).reset_index(drop=True)

# ————— 6) Features numéricas extra —————
def compute_features(texts: pd.Series) -> np.ndarray:
    lengths     = texts.str.len().to_numpy().reshape(-1,1)
    upper_ratio = texts.str.count(r"[A-Z]").to_numpy().reshape(-1,1) / (lengths + 1)
    return np.hstack([lengths, upper_ratio])

# ————— 7) Pipeline completo —————
def main():
    df = build_dataset()
    df = augment_and_oversample(df, llm_limit=300)
    print(f"Total tras oversample: {len(df)}")

    X, y = df["aug_text"], df["label"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    vec        = TfidfVectorizer(stop_words=SPANISH_STOPWORDS,
                                 max_features=5000, ngram_range=(1,2))
    X_tr_tf    = vec.fit_transform(X_tr)
    X_te_tf    = vec.transform(X_te)
    feat_tr    = compute_features(X_tr)
    feat_te    = compute_features(X_te)
    X_tr_full  = csr_matrix(hstack([X_tr_tf, csr_matrix(feat_tr)]))
    X_te_full  = csr_matrix(hstack([X_te_tf, csr_matrix(feat_te)]))

    svc = LinearSVC(class_weight={0:1,1:2}, max_iter=10_000, random_state=42)
    cal = CalibratedClassifierCV(estimator=svc, cv=3, method="isotonic")
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
    print(classification_report(y_te, y_pred, digits=4))

    joblib.dump(cal,      "svm_enhanced_calibrated.pkl")
    joblib.dump(vec,      "tfidf_enhanced_vectorizer.pkl")
    joblib.dump(best_thr, "best_thr.pkl")
    print("Modelos y umbral guardados.")

if __name__ == "__main__":
    main()