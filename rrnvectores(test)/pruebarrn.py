import tensorflow as tf
import joblib
import re

model = tf.keras.models.load_model('rrnvectores/mlp_model.h5')
vectorizer = joblib.load('rrnvectores/tfidf_vectorizer2.pkl')

def limpiar_texto(texto):
    texto = re.sub(r"http\S+", "", texto)
    texto = re.sub(r"@\w+", "", texto)
    texto = re.sub(r"[^\w\s]", "", texto)
    return texto.lower().strip()

def predecir(texto):
    texto_limpio = limpiar_texto(texto)
    vector = vectorizer.transform([texto_limpio]).toarray()
    prob = model.predict(vector)[0][0]
    etiqueta = "Falsa" if prob > 0.5 else "Verdadera"
    print(f"Predicción: {etiqueta} con probabilidad {prob:.4f}")

# Ejemplo
predecir("Ministra argentina Bullrich destaca la captura de alias ‘Fito’ en una reunión con la embajadora Diana Salazar")
