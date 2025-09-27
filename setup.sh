#!/bin/bash
# setup.sh - Script de instalación para Streamlit Cloud

echo "🔧 Instalando dependencias..."

# Instalar pip si no existe
python -m ensurepip --upgrade

# Actualizar pip
python -m pip install --upgrade pip

# Instalar dependencias
pip install -r requirements.txt

# Verificar instalación
echo "✅ Dependencias instaladas:"
pip list | grep -E "(streamlit|scikit-learn|joblib|pandas)"

echo "🚀 Configuración completada!"
