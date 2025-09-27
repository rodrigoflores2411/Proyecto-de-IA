# main.py - Actualizado para Kaggle Hub

import pandas as pd
import os
import sys

# Añadir el directorio raíz al path para imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.loader import DataLoader
from data.preprocessor import DataPreprocessor
from data.explorer import DataExplorer
from models.trainer import ModelTrainer
from models.evaluator import ModelEvaluator
from config import MODEL_SAVE_PATH, PLOT_SAVE_PATH, DATASET_SOURCES

def main():
    """Función principal que ejecuta el pipeline completo"""
    print("🩺 INICIANDO SISTEMA DE PREDICCIÓN DE DIABETES")
    print("=" * 50)
    
    # Paso 1: Cargar datos desde Kaggle Hub
    print("\n1️⃣  CARGANDO DATOS DESDE KAGGLE HUB...")
    loader = DataLoader()
    
    # Intentar cargar desde Kaggle, con fallback a URL
    df = loader.load_data(source='kaggle')
    
    if df is None:
        print("❌ Error: No se pudieron cargar los datos")
        return
    
    # Guardar copia local para futuras ejecuciones
    loader.save_local_copy()
    
    # Mostrar información del dataset
    info = loader.get_data_info()
    print(f"📊 Dataset cargado: {info['shape'][0]} registros, {info['shape'][1]} características")
    print(f"📁 Fuente: {info['source_path']}")
    
    # Paso 2: Análisis exploratorio
    print("\n2️⃣  ANÁLISIS EXPLORATORIO...")
    explorer = DataExplorer(df)
    report = explorer.generate_summary_report()
    
    # Paso 3: Preprocesamiento
    print("\n3️⃣  PREPROCESAMIENTO DE DATOS...")
    preprocessor = DataPreprocessor()
    splits = preprocessor.preprocess(df)
    
    # Resto del código se mantiene igual...
    # Paso 4: Entrenamiento de modelos
    print("\n4️⃣  ENTRENAMIENTO DE MODELOS...")
    trainer = ModelTrainer()
    trainer.initialize_models()
    training_results = trainer.train_models(splits)
    
    # Mostrar resumen del entrenamiento
    summary = trainer.get_training_summary()
    print("\n📊 RESUMEN DEL ENTRENAMIENTO:")
    print(summary.to_string(index=False))
    
    # Paso 5: Evaluación de modelos
    print("\n5️⃣  EVALUACIÓN DE MODELOS...")
    evaluator = ModelEvaluator(training_results, splits)
    test_results = evaluator.evaluate_on_test()
    
    # Generar reportes comparativos
    comparison_df = evaluator.generate_comparison_report()
    print("\n📈 COMPARACIÓN EN CONJUNTO DE PRUEBA:")
    print(comparison_df.to_string(index=False))
    
    # Generar gráficos de evaluación
    evaluator.plot_confusion_matrices()
    evaluator.plot_roc_curves()
    evaluator.plot_metrics_comparison()
    
    # Paso 6: Guardar modelos
    print("\n6️⃣  GUARDANDO MODELOS...")
    trainer.save_models()
    
    # Paso 7: Resumen final
    print("\n" + "=" * 50)
    print("🎉 PIPELINE COMPLETADO EXITOSAMENTE")
    print("=" * 50)
    
    best_model_test = comparison_df.loc[comparison_df['AUC-ROC'].idxmax()]
    print(f"🏆 MEJOR MODELO: {best_model_test['Modelo']}")
    print(f"📊 MÉTRICAS EN TEST:")
    print(f"   - AUC-ROC: {best_model_test['AUC-ROC']}")
    print(f"   - Accuracy: {best_model_test['Accuracy']}")
    print(f"   - F1-Score: {best_model_test['F1-Score']}")
    
    print(f"\n💾 Modelos guardados en: {MODEL_SAVE_PATH}")
    print(f"📊 Gráficos guardados en: {PLOT_SAVE_PATH}")

if __name__ == "__main__":
    main()