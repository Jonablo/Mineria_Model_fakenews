import streamlit as st
import joblib
import numpy as np

# 1️⃣ Carga de modelos y parámetros
model      = joblib.load('vectores/svm_fallback_model.pkl')
vectorizer = joblib.load('vectores/tfidf_fallback_vectorizer.pkl')

# Intentamos cargar el umbral óptimo (mejor F1)
try:
    best_thr = joblib.load('vectores/best_thr.pkl')
except FileNotFoundError:
    # Valor por defecto si no existe el fichero (reemplaza si guardas otro nombre)
    best_thr = 0.5

st.title("Detector de Noticias Falsas")
st.write("Ingresa una noticia y el modelo te dirá la probabilidad de que sea verdadera o falsa.")

# 2️⃣ Input de texto por usuario
texto_usuario = st.text_area("Ingresa la noticia aquí:", height=150)

if st.button("Predecir"):
    if not texto_usuario.strip():
        st.warning("Por favor, ingresa un texto para predecir.")
    else:
        # 3️⃣ Vectorizar y chequear que haya características
        X_new_tfidf = vectorizer.transform([texto_usuario])
        if X_new_tfidf.nnz == 0:
            st.warning(
                "Tu noticia no contiene ninguno de los términos vistos en el entrenamiento. "
                "Intenta reescribirla o verifícala."
            )
        # 4️⃣ Obtener probabilidades
        proba = model.predict_proba(X_new_tfidf)[0]
        proba_true = proba[0]
        proba_fake = proba[1]
        st.write(f"- Prob Verdadera: {proba_true*100:.2f}%")
        st.write(f"- Prob Falsa    : {proba_fake*100:.2f}%")

        # 5️⃣ Decisión usando umbral óptimo
        etiqueta = "Falsa" if proba_fake >= best_thr else "Verdadera"
        st.markdown(f"## Predicción: **{etiqueta}**")
        st.markdown(f"**(Umbral aplicado: {best_thr:.2f})**")

        # 6️⃣ Gráfico de pastel
        import matplotlib.pyplot as plt

        etiquetas = ['Verdadera', 'Falsa']
        valores   = [proba_true, proba_fake]
        fig, ax = plt.subplots()
        ax.pie(valores, labels=etiquetas, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)
