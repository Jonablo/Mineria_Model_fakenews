import streamlit as st
import joblib
import numpy as np

# Cargar modelo y vectorizador guardados
model = joblib.load('svm_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

st.title("Detector de Noticias Falsas")
st.write("Ingresa una noticia y el modelo te dirá la probabilidad de que sea verdadera o falsa.")

# Input de texto por usuario
texto_usuario = st.text_area("Ingresa la noticia aquí:", height=150)

if st.button("Predecir"):
    if not texto_usuario.strip():
        st.warning("Por favor, ingresa un texto para predecir.")
    else:
        # Vectorizar
        X_new_tfidf = vectorizer.transform([texto_usuario])

        # Predecir
        pred = model.predict(X_new_tfidf)[0]
        proba = model.predict_proba(X_new_tfidf)[0]

        # Etiqueta
        etiqueta = "Falsa" if pred == 1 else "Verdadera"

        st.markdown(f"### Predicción: {etiqueta}")
        st.markdown(f"### Probabilidades:")
        st.write(f"- Verdadera: {proba[0]*100:.2f}%")
        st.write(f"- Falsa: {proba[1]*100:.2f}%")

        # Gráfico de pastel con matplotlib para probabilidades
        import matplotlib.pyplot as plt

        etiquetas = ['Verdadera', 'Falsa']
        valores = proba
        colores = ['#4CAF50', '#F44336']

        fig, ax = plt.subplots()
        ax.pie(valores, labels=etiquetas, autopct='%1.1f%%', startangle=90, colors=colores)
        ax.axis('equal')
        st.pyplot(fig)
