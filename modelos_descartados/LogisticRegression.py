import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib

#1. Cargar los datos
df_train = pl.read_csv("datos/train.csv")
df_test = pl.read_csv("datos/test.csv")

#2. Convertir a pandas para usar con scikit-learn
df_train_pd = df_train.to_pandas()
df_test_pd = df_test.to_pandas()

#3. Preprocesamiento
X_train = df_train_pd['text']
y_train = df_train_pd['label']

X_test = df_test_pd['text']
y_test = df_test_pd['label']

#4. Vectorización TF-IDF
vectorizer = TfidfVectorizer(stop_words='spanish', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

#5. Modelo
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

#6. Evaluación
y_pred = model.predict(X_test_tfidf)
print("Reporte de clasificación:")
print(classification_report(y_test, y_pred))
print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))

#7. Guardar modelo y vectorizador
joblib.dump(model, "modelo_fakenews.pkl")
joblib.dump(vectorizer, "vectorizador_tfidf.pkl")