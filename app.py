import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Diabetes Prediction",
    layout="centered",  
    initial_sidebar_state="collapsed"
)

st.title("Predicción de Diabetes Tipo 2")
st.write("Sistema basado en Machine Learning")
age = st.slider("Edad", 20, 80, 45)
glucose = st.slider("Glucosa", 70, 200, 100)
bmi = st.slider("BMI", 18.0, 40.0, 25.0)

if st.button("Calcular Riesgo"):
    # Cálculo simple de ejemplo 
    risk_score = (glucose - 70) / 130 * 0.5 + (bmi - 18) / 22 * 0.3 + (age - 20) / 60 * 0.2
    risk_percentage = min(risk_score * 100, 95)
    
    st.progress(risk_percentage / 100)
    st.write(f"Riesgo estimado: {risk_percentage:.1f}%")
    
    if risk_percentage < 30:
        st.success("Riesgo bajo")
    elif risk_percentage < 70:
        st.warning("Riesgo moderado")
    else:
        st.error("Riesgo alto")

st.info("Esta es una versión simplificada para evitar errores de interfaz.")
