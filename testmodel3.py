import joblib

# Cargar modelo y vectorizador guardados
model = joblib.load('svm_balanced_calibrated.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Ejemplo de textos para predecir
nuevos_textos = [
    "Consejo Nacional Electoral se aprestan a quitar la multa solo en recintos donde Noboa perdió por más de 10 puntos, denuncia @ecuarauz, tras pedido del Gobierno de no multar a personas que no vayan a votar.",
    "Según el articulo 127 del Código de la Democracia, un acta que no tiene firmas conjuntas de Presidente y Secretario de la Junta, no tiene validez.",
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
