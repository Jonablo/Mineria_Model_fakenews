import joblib

# Cargar modelo y vectorizador guardados
model = joblib.load('svm_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Ejemplo de textos para predecir
nuevos_textos = [
    "En un evento realizado hoy en Quito, el Ministerio de Transporte y Obras Públicas anunció un plan de inversión sin precedentes de 1.2 mil millones de dólares destinado a la rehabilitación y construcción de carreteras y puentes en todo el territorio nacional.",
    "México rechaza vinculación con supuestos actos delictivos en Ecuador",
    "Por el caso Narcotentáculos, el Consejo de la Judicatura suspendió a cinco funcionarios judiciales de Manabí, cuyos domicilios y oficinas fueron allanados."
]

# Vectorizar los textos nuevos
X_new_tfidf = vectorizer.transform(nuevos_textos)

# Predecir con el modelo
predicciones = model.predict(X_new_tfidf)
probabilidades = model.predict_proba(X_new_tfidf)

for texto, pred, prob in zip(nuevos_textos, predicciones, probabilidades):
    etiqueta = "Falsa" if pred == 1 else "Verdadera"
    print(f"Noticia: {texto}\nPredicción: {etiqueta} (Probabilidades: {prob})\n")
