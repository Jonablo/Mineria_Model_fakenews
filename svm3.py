import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
import joblib 

# Descargar stopwords español si no está
nltk.download('stopwords')
spanish_stopwords = stopwords.words('spanish')

# Cargar datos
df_train = pd.read_csv("datos/train.csv")
df_fakes = pd.read_csv("datos/onlyfakes1000.csv")
df_true = pd.read_csv("noticias_rss_espanol.csv")

# Asignar etiquetas a datasets sin label
df_fakes['label'] = 1
df_true['label'] = 0

# Revisar columna que tiene el texto (asumo 'text' pero confirma si es otro)
print(df_train.columns)
print(df_fakes.columns)
print(df_true.columns)

# Seleccionar solo columna de texto y label (por si hay columnas extra)
df_fakes = df_fakes[['text', 'label']]
df_true = df_true[['text', 'label']]
df_train = df_train[['text', 'label']]

# Concatenar
df_all = pd.concat([df_train, df_fakes, df_true], ignore_index=True)

# Shuffle
df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    df_all['text'], df_all['label'], test_size=0.2, random_state=42, stratify=df_all['label']
)

# Vectorización TF-IDF
vectorizer = TfidfVectorizer(stop_words=spanish_stopwords, max_features=5000, ngram_range=(1,2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Entrenar SVM con parámetros estándar
model = SVC(C=1, kernel='linear', probability=True, random_state=42)
model.fit(X_train_tfidf, y_train)

# Predecir y evaluar
y_pred = model.predict(X_test_tfidf)
print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

# Guardar modelo y vectorizador entrenados
joblib.dump(model, 'svm_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')