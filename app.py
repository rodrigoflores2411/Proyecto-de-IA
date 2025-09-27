import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
import os

# Añadir rutas para imports - CORREGIDO
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from config import MODEL_SAVE_PATH, RISK_THRESHOLDS, FEATURE_NAMES

# Configuración de la página
st.set_page_config(
    page_title="Predicción de Diabetes Tipo 2",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Título principal
st.title("🩺 Sistema de Predicción de Riesgo de Diabetes Tipo 2")
st.markdown("---")

# Sidebar con información
st.sidebar.title("Información del Sistema")
st.sidebar.info("""
**Características del modelo:**
- ✅ Random Forest, XGBoost y Redes Neuronales
- ✅ Precisión: >75% (AUC > 0.75)
- ✅ Datos clínicos validados
- ✅ Interfaz médica intuitiva
""")

# Cargar modelo y escalador
@st.cache_resource
def load_model():
    try:
        model = joblib.load(f'{MODEL_SAVE_PATH}best_model.pkl')
        scaler = joblib.load(f'{MODEL_SAVE_PATH}scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"Error cargando el modelo: {e}")
        return None, None

model, scaler = load_model()

# Si el modelo no está cargado, mostrar opción para entrenar
if model is None:
    st.warning("⚠️ Modelo no encontrado. Por favor, entrena el modelo primero.")
    
    if st.button("🎯 Entrenar Modelo (Ejecutar Pipeline)"):
        with st.spinner("Entrenando modelo... Esto puede tomar unos minutos"):
            try:
                from main import main
                main()
                st.success("✅ Modelo entrenado exitosamente!")
                st.rerun()
            except Exception as e:
                st.error(f"Error entrenando modelo: {e}")
    
    st.stop()

# Interfaz principal de predicción
st.header("🎯 Predicción de Riesgo Individual")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Datos Clínicos del Paciente")
    
    pregnancies = st.slider("Número de embarazos", 0, 20, 1,
                           help="Número total de embarazos")
    
    glucose = st.slider("Glucosa plasmática (mg/dL)", 0, 200, 100,
                       help="Concentración de glucosa a 2 horas")
    
    blood_pressure = st.slider("Presión arterial (mm Hg)", 0, 150, 70,
                              help="Presión arterial diastólica")
    
    skin_thickness = st.slider("Espesor pliegue cutáneo (mm)", 0, 100, 20,
                              help="Espesor del pliegue cutáneo del tríceps")

with col2:
    st.subheader(" ")
    st.write("")  # Espaciador
    
    insulin = st.slider("Insulina sérica (mu U/ml)", 0, 900, 80,
                       help="Insulina sérica a 2 horas")
    
    bmi = st.slider("Índice de masa corporal (kg/m²)", 0.0, 70.0, 25.0, 0.1,
                   help="BMI calculado como peso/(altura²)")
    
    diabetes_pedigree = st.slider("Función de pedigrí diabetes", 0.0, 2.5, 0.5, 0.01,
                                 help="Funcióń que resume historia familiar")
    
    age = st.slider("Edad (años)", 0, 120, 30,
                   help="Edad de la paciente")

# Botón de predicción
if st.button("🔍 Predecir Riesgo", type="primary", use_container_width=True):
    # Preparar datos de entrada
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                          insulin, bmi, diabetes_pedigree, age]])
    
    # Escalar datos
    input_scaled = scaler.transform(input_data)
    
    # Realizar predicción
    try:
        probability = model.predict_proba(input_scaled)[0][1]
        
        # Mostrar resultados
        st.success("✅ Predicción completada")
        
        # Métricas visuales
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Probabilidad de Diabetes", f"{probability:.2%}")
        
        with col2:
            # Barra de progreso
            st.write("Nivel de Riesgo")
            st.progress(float(probability))
        
        with col3:
            # Indicador de riesgo
            if probability < RISK_THRESHOLDS['low']:
                risk_level = "BAJO"
                risk_color = "🟢"
            elif probability < RISK_THRESHOLDS['medium']:
                risk_level = "MODERADO"
                risk_color = "🟡"
            else:
                risk_level = "ALTO"
                risk_color = "🔴"
            
            st.metric("Nivel de Riesgo", f"{risk_color} {risk_level}")
        
        # Recomendaciones
        st.subheader("💡 Recomendaciones Médicas")
        
        if risk_level == "BAJO":
            st.info("""
            **Riesgo Bajo - Mantener prevención:**
            - Continuar con estilo de vida saludable
            - Control anual de glucosa en sangre
            - Mantener peso adecuado y actividad física regular
            """)
        elif risk_level == "MODERADO":
            st.warning("""
            **Riesgo Moderado - Vigilancia activa:**
            - Consultar con médico para evaluación completa
            - Realizar prueba de tolerancia a la glucosa
            - Implementar cambios en dieta y ejercicio
            - Control trimestral de parámetros
            """)
        else:
            st.error("""
            **Riesgo Alto - Acción inmediata:**
            - Consulta médica URGENTE
            - Realizar pruebas diagnósticas completas
            - Implementar plan de tratamiento supervisado
            - Control mensual estricto
            """)
        
        # Detalles técnicos (expandible)
        with st.expander("📊 Detalles Técnicos de la Predicción"):
            st.write(f"**Probabilidad de NO diabetes:** {1-probability:.2%}")
            st.write(f"**Probabilidad de diabetes:** {probability:.2%}")
            st.write(f"**Umbral de clasificación:** 0.5")
            st.write(f"**Modelo utilizado:** {type(model).__name__}")
            
            # Características ingresadas
            st.write("**Datos ingresados:**")
            feature_values = {
                'Embarazos': pregnancies,
                'Glucosa': f"{glucose} mg/dL",
                'Presión Arterial': f"{blood_pressure} mm Hg",
                'Pliegue Cutáneo': f"{skin_thickness} mm",
                'Insulina': f"{insulin} mu U/ml",
                'BMI': f"{bmi} kg/m²",
                'Pedigrí Diabetes': diabetes_pedigree,
                'Edad': f"{age} años"
            }
            
            for feature, value in feature_values.items():
                st.write(f"- {feature}: {value}")
    
    except Exception as e:
        st.error(f"❌ Error en la predicción: {e}")

# Sección de lote (opcional)
st.markdown("---")
st.header("📁 Predicción por Lote")

uploaded_file = st.file_uploader("Subir archivo CSV con datos de pacientes", 
                                type=['csv'])

if uploaded_file is not None:
    try:
        # Leer archivo
        batch_data = pd.read_csv(uploaded_file)
        
        # Verificar columnas
        required_cols = FEATURE_NAMES
        if all(col in batch_data.columns for col in required_cols):
            st.success(f"✅ Archivo cargado: {len(batch_data)} pacientes")
            
            # Preprocesar y predecir
            X_batch = batch_data[required_cols]
            X_batch_scaled = scaler.transform(X_batch)
            probabilities = model.predict_proba(X_batch_scaled)[:, 1]
            
            # Añadir resultados al DataFrame
            results_df = batch_data.copy()
            results_df['Probabilidad_Diabetes'] = probabilities
            results_df['Riesgo'] = results_df['Probabilidad_Diabetes'].apply(
                lambda p: 'BAJO' if p < 0.3 else 'MODERADO' if p < 0.7 else 'ALTO'
            )
            
            # Mostrar resultados
            st.subheader("Resultados del Lote")
            st.dataframe(results_df)
            
            # Estadísticas
            risk_counts = results_df['Riesgo'].value_counts()
            st.write("**Distribución de riesgos:**")
            for riesgo, count in risk_counts.items():
                st.write(f"- {riesgo}: {count} pacientes ({count/len(results_df)*100:.1f}%)")
            
            # Descargar resultados
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Descargar Resultados en CSV",
                data=csv,
                file_name="resultados_diabetes.csv",
                mime="text/csv"
            )
            
        else:
            st.error(f"❌ El archivo debe contener las columnas: {', '.join(required_cols)}")
    
    except Exception as e:
        st.error(f"❌ Error procesando archivo: {e}")

# Footer
st.markdown("---")
st.markdown(
    "**Proyecto de Inteligencia Artificial - Universidad Privada Antenor Orrego** • "
    "Integrantes: Flores Alvarez, Moreno Rodríguez, Soto Gonzales"
)