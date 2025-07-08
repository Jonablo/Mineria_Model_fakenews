import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go # type: ignore

# ——— Configuración de página ———
st.set_page_config(
    page_title="Detector de Noticias Falsas",
    page_icon="📰",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ——— CSS a medida ———
st.markdown("""
<style>
/* Fondo general */
body {
    background-color: #f0f2f6;
    color: #222222;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* Título */
[data-testid="stHeader"] h1 {
    color: #0d1b2a;
    font-size: 2.5rem;
    margin-bottom: 0;
}

/* Separadores */
hr {
    border: none;
    height: 1px;
    background-color: #d3d3d3;
    margin: 1.5rem 0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #ffffff;
    padding: 1rem;
    border-right: 1px solid #d3d3d3;
}

/* Área de texto */
textarea {
    font-size: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ——— Header con logo opcional ———
col_logo, col_header = st.columns([1, 9])
with col_header:
    st.title("Detector de Noticias Falsas")
    st.markdown("**SVM + TF-IDF | Clasificación binaria de noticias**")

st.markdown("---")

# ——— Input del usuario ———
st.subheader("Ingresar nota para evaluar")
texto = st.text_area("", height=180)

# ——— Lógica de predicción ———
if st.button("Evaluar noticia"):
    if not texto.strip():
        st.warning("Debe ingresar algún texto antes de evaluar.")
    else:
        # Carga de modelo (cache para no recargar cada vez)
        @st.cache_resource
        def load_artifacts():
            m = joblib.load("svm_model.pkl")
            v = joblib.load("tfidf_vectorizer.pkl")
            return m, v

        model, vectorizer = load_artifacts()

        # Vectorizar y predecir
        X = vectorizer.transform([texto])
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0]

        p_true = proba[0] * 100
        p_fake = proba[1] * 100
        etiqueta = "Verdadera" if pred == 0 else "Falsa"

        # ——— Resultados en métricas ———
        m1, m2, m3 = st.columns(3)
        m1.metric(label="Clasificación", value=etiqueta)
        m2.metric(label="Prob. Verdadera", value=f"{p_true:.2f}%")
        m3.metric(label="Prob. Falsa", value=f"{p_fake:.2f}%")

        st.markdown("")  # pequeño espacio

        # ——— Gráfico profesional con Plotly ———
        fig = go.Figure(
            go.Pie(
                labels=["Verdadera", "Falsa"],
                values=[p_true, p_fake],
                hole=0.6,
                marker=dict(colors=["#0d1b2a", "#ba0c2f"]),
                sort=False,
                direction="clockwise"
            )
        )
        fig.update_traces(
            texttemplate="%{label}<br>%{percent}",
            textfont=dict(size=14, color="#ffffff"),
            showlegend=False
        )
        fig.update_layout(
            margin=dict(l=0, r=0, t=30, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            annotations=[dict(
                text="Probabilidades",
                x=0.5, y=0.5,
                font_size=16,
                showarrow=False,
                font_color="#222222"
            )]
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.info("**Nota:** Este sistema es una herramienta de apoyo. Siempre verifique con fuentes oficiales.")
