import os
import joblib
import streamlit as st
import plotly.express as px
import numpy as np
from scipy.sparse import hstack, csr_matrix

# ————— Page & theme config —————
st.set_page_config(
    page_title="Detector de Noticias Falsas",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ————— Custom CSS for cards & buttons —————
st.markdown(
    """
    <style>
    .stMetric > div {
        background: #1e1e2e !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3) !important;
        padding: 10px !important;
    }
    .stButton>button {
        background-color: #ff7f0e !important;
        color: white !important;
        border: none !important;
        padding: 0.6em 1.2em !important;
        font-size: 1.0em !important;
        border-radius: 6px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ————— Sidebar controls —————
st.sidebar.header("⚙️ Configuración")
try:
    stored_thr = joblib.load("best_thr_sin_oversample.pkl")
except FileNotFoundError:
    stored_thr = 0.528
best_thr = st.sidebar.slider(
    "Umbral mínimo para Falsa", 0.0, 1.0, float(stored_thr), 0.01
)

# ————— Load model + vectorizer —————
@st.cache_resource
def load_artifacts():
    model = joblib.load("svm_sin_oversample.pkl")
    vect  = joblib.load("tfidf_sin_oversample_vectorizer.pkl")
    return model, vect

model, vectorizer = load_artifacts()

# ————— Feature helper —————
def compute_features(text: str) -> np.ndarray:
    length = len(text)
    upper_count = sum(1 for c in text if c.isupper())
    upper_ratio = upper_count / (length + 1)
    return np.array([[length, upper_ratio]])

# ————— App title & input —————
st.title("📰 Detector de Noticias Falsas")
st.write(
    "Ingresa el texto de una noticia y haz clic en **Predecir** para ver "
    "la probabilidad de que sea verdadera o falsa."
)

texto_usuario = st.text_area("📝 Noticia:", height=200)

# ————— Prediction button —————
if st.button("🔍 Predecir"):
    if not texto_usuario.strip():
        st.warning("Por favor, ingresa algún texto antes de predecir.")
    else:
        # 1) TF-IDF vectorization
        X_tfidf = vectorizer.transform([texto_usuario])
        if X_tfidf.nnz == 0:
            st.warning(
                "El texto no contiene ninguno de los términos vistos en entrenamiento."
            )
        # 2) numeric features
        feat = compute_features(texto_usuario)
        X_new = hstack([X_tfidf, csr_matrix(feat)])
        # 3) predict probabilities
        p_true, p_fake = model.predict_proba(X_new)[0]

        # — Metric cards —
        c1, c2 = st.columns(2, gap="large")
        c1.metric("Probabilidad Verdadera", f"{p_true*100: .1f}%")
        c2.metric("Probabilidad Falsa",      f"{p_fake*100: .1f}%")

        # — Pie chart —
        fig = px.pie(
            names=["Verdadera", "Falsa"],
            values=[p_true, p_fake],
            hole=0.4,
            color_discrete_sequence=["#1f77b4", "#ff7f0e"]
        )
        fig.update_layout(
            margin=dict(t=0, b=0, l=0, r=0),
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)

        # — Final label —
        etiqueta = "🔴 Falsa" if p_fake >= best_thr else "🟢 Verdadera"
        st.markdown(f"## Predicción final: **{etiqueta}**")
        st.markdown(f"_Umbral aplicado: {best_thr:.2f}_")