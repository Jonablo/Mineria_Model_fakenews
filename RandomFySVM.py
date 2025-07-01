import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
import joblib

# Descargar stopwords
nltk.download('stopwords')
spanish_stopwords = stopwords.words('spanish')

# Cargar dataset
df = pd.read_csv("datos/train.csv")

# Dividir en train y test
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

# Vectorización TF-IDF
vectorizer = TfidfVectorizer(stop_words=spanish_stopwords, max_features=5000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Modelo Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_tfidf, y_train)
y_pred_rf = rf.predict(X_test_tfidf)

print("=== Random Forest ===")
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# Modelo SVM
#(kernels: linear, poly, rbf, sigmoid)
svm = SVC(kernel='linear', probability=True, random_state=42)
svm.fit(X_train_tfidf, y_train)
y_pred_svm = svm.predict(X_test_tfidf)

print("=== Support Vector Machine ===")
print(confusion_matrix(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))

# Guardar los modelos y vectorizador
joblib.dump(rf, "modelo_rf.pkl")
joblib.dump(svm, "modelo_svm.pkl")
joblib.dump(vectorizer, "vectorizador_tfidf.pkl")