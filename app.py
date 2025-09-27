# app.py - VERSIÓN MÍNIMA PARA DIAGNÓSTICO

import streamlit as st
import sys
import os

# Añadir el directorio actual al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

st.title("🩺 Prueba de Dependencias - Diabetes Prediction")
st.markdown("---")

# Probar importaciones básicas
st.header("1. Probando importaciones básicas")

try:
    import pandas as pd
    st.success("✅ pandas importado correctamente")
    st.write(f"Versión pandas: {pd.__version__}")
except ImportError as e:
    st.error(f"❌ Error importando pandas: {e}")

try:
    import numpy as np
    st.success("✅ numpy importado correctamente")
    st.write(f"Versión numpy: {np.__version__}")
except ImportError as e:
    st.error(f"❌ Error importando numpy: {e}")

try:
    import joblib
    st.success("✅ joblib importado correctamente")
    st.write(f"Versión joblib: {joblib.__version__}")
except ImportError as e:
    st.error(f"❌ Error importando joblib: {e}")

# Probar importaciones de ML
st.header("2. Probando importaciones de ML")

try:
    from sklearn.ensemble import RandomForestClassifier
    st.success("✅ scikit-learn importado correctamente")
except ImportError as e:
    st.error(f"❌ Error importando scikit-learn: {e}")

try:
    import xgboost as xgb
    st.success("✅ xgboost importado correctamente")
except ImportError as e:
    st.error(f"❌ Error importando xgboost: {e}")

# Probar nuestras propias importaciones
st.header("3. Probando importaciones personalizadas")

try:
    # Intentar importar config
    from config import TARGET_NAME, FEATURE_NAMES
    st.success("✅ config.py importado correctamente")
    st.write(f"Target: {TARGET_NAME}")
    st.write(f"Features: {FEATURE_NAMES}")
except Exception as e:
    st.error(f"❌ Error importando config: {e}")

try:
    # Intentar importar helpers
    from utils.helpers import get_risk_recommendations
    st.success("✅ helpers.py importado correctamente")
    
    # Probar función
    risk_info = get_risk_recommendations(0.5)
    st.write(f"Recomendación de prueba: {risk_info['level']}")
except Exception as e:
    st.error(f"❌ Error importando helpers: {e}")

# Información del sistema
st.header("4. Información del sistema")

st.write(f"Directorio de trabajo: {os.getcwd()}")
st.write(f"Archivos en directorio: {os.listdir('.')}")

if os.path.exists('requirements.txt'):
    with open('requirements.txt', 'r') as f:
        requirements = f.read()
    st.text_area("Contenido de requirements.txt:", requirements, height=200)
else:
    st.error("❌ requirements.txt no encontrado")

# Footer
st.markdown("---")
st.markdown("**App de diagnóstico - Si todas las importaciones son ✅, la app funciona**")
