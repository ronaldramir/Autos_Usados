
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Clasificación de vehículos usados (Clusters)",
    page_icon="🚗",
    layout="centered"
)

MODEL_PATH = Path("modelo_clasificacion_clusters.pkl")

cluster_labels = {
    0: "Premium europeo",
    1: "Familiares y trabajo",
    2: "Económicos masivos"
}

cluster_desc = {
    0: "Vehículos de perfil premium, típicamente europeos, con mayor precio y equipamiento.",
    1: "Vehículos orientados a familia y trabajo: espacio, practicidad y uso mixto.",
    2: "Vehículos masivos económicos: enfoque en costo, disponibilidad y mantenimiento accesible."
}

@st.cache_resource
def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo del modelo en: {model_path.resolve()}\n"
            "Colocá 'modelo_clasificacion_clusters.pkl' en la misma carpeta que app.py"
        )
    return joblib.load(model_path)

def build_input_df(
    precio_crc, kilometraje, antiguedad, cilindrada, puertas, pasajeros,
    marca_freq, premium_flag,
    estilo, combustible, transmision, segmento_marca, origen_marca
):
    data = {
        "precio_crc": [precio_crc],
        "kilometraje": [kilometraje],
        "antiguedad": [antiguedad],
        "cilindrada": [cilindrada],
        "puertas": [puertas],
        "pasajeros": [pasajeros],
        "marca_freq": [marca_freq],
        "premium_flag": [premium_flag],
        "estilo": [estilo],
        "combustible": [combustible],
        "transmision": [transmision],
        "segmento_marca": [segmento_marca],
        "origen_marca": [origen_marca],
    }
    return pd.DataFrame(data)

st.title("🚗 Clasificación de vehículos usados por segmento")
st.caption(
    "Este modelo asigna vehículos a segmentos existentes (no descubre nuevos). "
    "Usa un pipeline (preprocesamiento + Random Forest) guardado en .pkl."
)

try:
    pipeline = load_model(MODEL_PATH)
except Exception as e:
    st.error(str(e))
    st.stop()

st.subheader("1) Variables numéricas")

col1, col2 = st.columns(2)

with col1:
    precio_crc = st.number_input("precio_crc (₡)", min_value=0.0, value=8000000.0, step=50000.0)
    kilometraje = st.number_input("kilometraje", min_value=0, value=80000, step=1000)
    antiguedad = st.number_input("antiguedad (años)", min_value=0, value=8, step=1)
    cilindrada = st.number_input("cilindrada (cc)", min_value=0, value=2000, step=100)

with col2:
    puertas = st.number_input("puertas", min_value=1, max_value=6, value=4, step=1)
    pasajeros = st.number_input("pasajeros", min_value=1, max_value=10, value=5, step=1)
    marca_freq = st.number_input("marca_freq", min_value=0, value=1200, step=10)
    premium_flag = st.selectbox("premium_flag", options=[0, 1], index=0)

st.subheader("2) Variables categóricas")

DEFAULT_ESTILO = ["Sedán", "SUV", "Hatchback", "Pickup", "Coupé", "Wagon", "Van"]
DEFAULT_COMBUSTIBLE = ["Gasolina", "Diésel", "Híbrido", "Eléctrico"]
DEFAULT_TRANSMISION = ["Manual", "Automática", "CVT"]
DEFAULT_SEGMENTO_MARCA = ["Económica", "Generalista", "Premium"]
DEFAULT_ORIGEN_MARCA = ["Japón", "Corea", "Europa", "USA", "China", "India"]

c1, c2 = st.columns(2)
with c1:
    estilo = st.selectbox("estilo", DEFAULT_ESTILO)
    combustible = st.selectbox("combustible", DEFAULT_COMBUSTIBLE)
    transmision = st.selectbox("transmision", DEFAULT_TRANSMISION)

with c2:
    segmento_marca = st.selectbox("segmento_marca", DEFAULT_SEGMENTO_MARCA)
    origen_marca = st.selectbox("origen_marca", DEFAULT_ORIGEN_MARCA)

df_input = build_input_df(
    precio_crc, kilometraje, antiguedad, cilindrada, puertas, pasajeros,
    marca_freq, premium_flag,
    estilo, combustible, transmision, segmento_marca, origen_marca
)

st.subheader("3) Predicción")

if st.button("🔎 Clasificar vehículo", type="primary"):
    try:
        pred = int(pipeline.predict(df_input)[0])
        label = cluster_labels.get(pred, f"Cluster {pred}")
        st.success(f"Cluster asignado: {pred}")
        st.info(f"Segmento: {label}")

        desc = cluster_desc.get(pred)
        if desc:
            st.write(desc)

        with st.expander("Ver DataFrame enviado al modelo"):
            st.dataframe(df_input, use_container_width=True)

    except Exception as e:
        st.error(f"Error en predicción: {e}")

st.caption("Modelo: RandomForestClassifier + Pipeline (.pkl)")
