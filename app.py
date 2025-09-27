import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
import os

st.set_page_config(
    page_title="Predicción de Diabetes Tipo 2",
    page_icon="🩺",
    layout="wide"
)

# Título principal
st.title("🩺 Sistema de Predicción de Riesgo de Diabetes Tipo 2")
st.markdown("---")
# Sidebar
st.sidebar.title("Información")
st.sidebar.info("Sistema de ML para predicción de diabetes")
#Cargar el modelo
@st.cache_resource
def load_model():
    try:

        model_paths = [
            'results/models/best_model.pkl',
            'best_model.pkl',
            'model.pkl'
        ]
        
        scaler_paths = [
            'results/models/scaler.pkl',
            'scaler.pkl'
        ]
        
        model = None
        scaler = None
        
        # Buscar modelo
        for path in model_paths:
            if os.path.exists(path):
                model = joblib.load(path)
                st.sidebar.success(f"✅ Modelo cargado: {path}")
                break
        
        # Buscar scaler
        for path in scaler_paths:
            if os.path.exists(path):
                scaler = joblib.load(path)
                st.sidebar.success(f"✅ Scaler cargado: {path}")
                break
        
        return model, scaler
        
    except Exception as e:
        st.sidebar.error(f"❌ Error cargando modelo: {e}")
        return None, None

model, scaler = load_model()

# Modelo a entrenar
if model is None:
    st.warning("""
    ⚠️ **Modelo no encontrado**
    
    Para usar la aplicación, primero debes entrenar el modelo:
    1. Ejecuta `python main.py` localmente
    2. Sube los archivos `.pkl` generados a la carpeta `results/models/`
    3. Recarga esta aplicación
    """)
    
    st.info("""
    **Archivos necesarios:**
    - `results/models/best_model.pkl`
    - `results/models/scaler.pkl`
    """)
    
    # Opción para generar datos de ejemplo
    if st.button("🎯 Usar Datos de Ejemplo (Demo)"):
        st.session_state.demo_mode = True
        st.success("✅ Modo demo activado. Puedes probar la interfaz.")

# Interfaz de predicción
st.header("🎯 Predicción de Riesgo Individual")

# Definir características (evitar importar de config)
FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.slider("Número de embarazos", 0, 20, 1)
    glucose = st.slider("Glucosa plasmática (mg/dL)", 0, 200, 100)
    blood_pressure = st.slider("Presión arterial (mm Hg)", 0, 150, 70)
    skin_thickness = st.slider("Espesor pliegue cutáneo (mm)", 0, 100, 20)

with col2:
    insulin = st.slider("Insulina sérica (mu U/ml)", 0, 900, 80)
    bmi = st.slider("Índice de masa corporal (kg/m²)", 0.0, 70.0, 25.0, 0.1)
    diabetes_pedigree = st.slider("Función de pedigrí diabetes", 0.0, 2.5, 0.5, 0.01)
    age = st.slider("Edad (años)", 0, 120, 30)

# Botón de predicción
if st.button("🔍 Predecir Riesgo", type="primary"):
    
    if model is None and not st.session_state.get('demo_mode', False):
        st.error("❌ No hay modelo disponible para hacer predicciones")
        st.stop()
    
    # Preparar datos de entrada
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                          insulin, bmi, diabetes_pedigree, age]])
    
    try:
        if st.session_state.get('demo_mode', False):
            # Modo demo - simular predicción
            probability = 0.45  # Valor de ejemplo
            st.info("🔶 **MODO DEMO**: Usando datos de ejemplo")
        else:
            # Escalar y predecir
            input_scaled = scaler.transform(input_data)
            probability = model.predict_proba(input_scaled)[0][1]
        
        # Mostrar resultados
        st.success("✅ Predicción completada")
        
        # Métricas visuales
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Probabilidad de Diabetes", f"{probability:.2%}")
        
        with col2:
            st.write("Nivel de Riesgo")
            st.progress(float(probability))
        
        with col3:
            if probability < 0.3:
                risk_level = "BAJO"
                risk_color = "🟢"
            elif probability < 0.7:
                risk_level = "MODERADO"
                risk_color = "🟡"
            else:
                risk_level = "ALTO"
                risk_color = "🔴"
            
            st.metric("Nivel de Riesgo", f"{risk_color} {risk_level}")
        
        # Recomendaciones
        st.subheader("💡 Recomendaciones")
        
        if risk_level == "BAJO":
            st.info("**Riesgo Bajo** - Mantener estilo de vida saludable")
        elif risk_level == "MODERADO":
            st.warning("**Riesgo Moderado** - Consultar con médico")
        else:
            st.error("**Riesgo Alto** - Acción médica inmediata")
            
    except Exception as e:
        st.error(f"❌ Error en la predicción: {str(e)}")

# Información adicional
with st.expander("📊 Información Técnica"):
    st.write("""
    **Características del sistema:**
    - Algoritmos: Random Forest, XGBoost, Redes Neuronales
    - Dataset: Pima Indians Diabetes Database
    - Métricas: Accuracy, Precision, Recall, F1-Score, AUC-ROC
    
    **Parámetros utilizados:**
    - Glucosa, Presión arterial, BMI, Edad, etc.
    """)
    
    if model is not None:
        st.write(f"**Modelo cargado:** {type(model).__name__}")

# Footer
st.markdown("---")
st.markdown(
    "**Proyecto de Inteligencia Artificial - Universidad Privada Antenor Orrego**"
)
