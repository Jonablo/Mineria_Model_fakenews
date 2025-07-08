import os
import openai
import pandas as pd
import nltk
from dotenv import load_dotenv
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import time

# Carga las variables de .env al entorno
load_dotenv()

# 1) Configuración de OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise RuntimeError("Define la variable OPENAI_API_KEY")

# 2) Stopwords
nltk.download("stopwords", quiet=True)
spanish_stopwords = stopwords.words("spanish")

# 3) Función de extracción con OpenAI
def extraer_hechos(texto: str) -> str:
    prompt = f"""
    Eres un extractor de hechos clave. De la noticia en español, extrae exactamente 3 frases breves, puras, sin numeración ni guiones, cada una con un solo hecho (un solo sujeto, un solo predicado). Nada más.

{texto}

Formato de salida:
Resumen: <frase>
Hechos: <hecho1>; <hecho2>; <hecho3>; <hecho4>; <hecho5>
"""
    resp = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user","content":prompt}],
        temperature=0.2,
        max_tokens=200
    )
    
    # Extraer contenido seguro
    contenido = resp.choices[0].message.content or ""
    # Ahora nunca es None
    return contenido.replace("\n"," ").strip()

# 4) Cargar y concatenar datasets
df_train = pd.read_csv("datos/train.csv")
df_fakes = pd.read_csv("datos/onlyfakes1000.csv"); df_fakes["label"]=1
df_true  = pd.read_csv("datos/onlytrue1000.csv");  df_true["label"]=0

df_all = pd.concat([
    df_train[['text','label']],
    df_fakes[['text','label']],
    df_true[['text','label']]
], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

# 5) Fallback parcial: solo las primeras N con LLM
N = 300
hechos_cache = {}
augmented = []

for idx, txt in enumerate(df_all["text"]):
    if idx < N:
        # llamamos LLM y cacheamos
        if txt in hechos_cache:
            bloque = hechos_cache[txt]
        else:
            bloque = extraer_hechos(txt)
            hechos_cache[txt] = bloque
            time.sleep(1)  # tasa segura
        augmented.append(f"{txt} {bloque}")
    else:
        # más allá de N: sin aumento (solo texto original)
        augmented.append(txt)

df_all["augmented_text"] = augmented

# 6) Train/Test split
X = df_all["augmented_text"]
y = df_all["label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 7) TF-IDF
vectorizer = TfidfVectorizer(
    stop_words=spanish_stopwords,
    max_features=5000,
    ngram_range=(1,2)
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf  = vectorizer.transform(X_test)

# 8) SVM
model = SVC(C=1, kernel="linear", probability=True, random_state=42)
model.fit(X_train_tfidf, y_train)

# 9) Evaluación
y_pred = model.predict(X_test_tfidf)
print("\nMatriz de confusión:")
print(confusion_matrix(y_test, y_pred))
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred, digits=4))

# 10) Guardar
joblib.dump(model,      "svm_fallback_model.pkl")
joblib.dump(vectorizer, "tfidf_fallback_vectorizer.pkl")