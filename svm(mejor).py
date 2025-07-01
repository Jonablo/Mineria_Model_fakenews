import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

nltk.download('stopwords')
spanish_stopwords = stopwords.words('spanish')

# Cargar dataset
df = pd.read_csv("datos/train.csv")

# Limpieza básica
df = df.dropna(subset=['text', 'label'])
df = df[df['text'].str.strip().str.len() > 10]

X = df['text']
y = df['label']

# Separamos conjunto de prueba independiente (20%)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Vectorizador ajustado SOLO con entrenamiento completo (sin usar datos de prueba)
vectorizer = TfidfVectorizer(stop_words=spanish_stopwords, max_features=5000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train_full)
X_test_tfidf = vectorizer.transform(X_test)

# Validación cruzada estratificada para búsqueda hiperparámetros en entrenamiento
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

param_svm = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

param_log = {
    'C': [0.1, 1, 10]
}

# GridSearch SVM sobre entrenamiento
grid_svm = GridSearchCV(SVC(probability=True, random_state=42), param_svm, cv=cv, scoring='accuracy', n_jobs=-1)
grid_svm.fit(X_train_tfidf, y_train_full)
best_svm = grid_svm.best_estimator_

# GridSearch Logistic Regression sobre entrenamiento
grid_log = GridSearchCV(LogisticRegression(max_iter=2000, random_state=42), param_log, cv=cv, scoring='accuracy', n_jobs=-1)
grid_log.fit(X_train_tfidf, y_train_full)
best_log = grid_log.best_estimator_

# Ensamble VotingClassifier con los mejores estimadores
ensemble = VotingClassifier(
    estimators=[('svm', best_svm), ('logreg', best_log)],
    voting='soft',
    weights=[3, 1]
)

# Entrenamos ensamble con todo el entrenamiento (completo)
ensemble.fit(X_train_tfidf, y_train_full)

# Evaluación final sobre el conjunto de prueba
y_pred = ensemble.predict(X_test_tfidf)

print(f"Mejor SVM: {grid_svm.best_params_}")
print(f"Mejor Logistic Regression: {grid_log.best_params_}")
print("Matriz de confusión final:")
print(confusion_matrix(y_test, y_pred))
print("\nReporte de clasificación final:")
print(classification_report(y_test, y_pred))
print(f"Accuracy final: {accuracy_score(y_test, y_pred):.4f}")