import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
import joblib

# Descargar stopwords en español
nltk.download('stopwords')

# Cargar lista de stopwords en español
spanish_stopwords = stopwords.words('spanish')

# Carga el dataset
df = pd.read_csv("datos/train.csv")


# Divide datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

# Vectorización TF-IDF con stopwords en español
vectorizer = TfidfVectorizer(stop_words=spanish_stopwords, max_features=5000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Entrenar modelo de regresión logística
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Predecir en conjunto de prueba
y_pred = model.predict(X_test_tfidf)

# Evaluar resultados
print("\nMatriz de confusión:")
print(confusion_matrix(y_test, y_pred))
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

# Guardar modelo y vectorizador
joblib.dump(model, "modelo_fakenews.pkl")
joblib.dump(vectorizer, "vectorizador_tfidf.pkl")