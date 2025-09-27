import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split  # IMPORT CORREGIDO
from sklearn.preprocessing import StandardScaler     # IMPORT CORREGIDO
import joblib
import os
import sys

# Añadir path para importar config
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config import PREPROCESSING_CONFIG, TARGET_NAME, MODEL_SAVE_PATH  # IMPORTS CORREGIDOS

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def preprocess(self, df):
        """
        Realiza todo el pipeline de preprocesamiento
        """
        # Crear copia para no modificar original
        data = df.copy()
        
        # 1. Limpieza de datos
        data_clean = self._clean_data(data)
        
        # 2. División en características y objetivo
        X, y = self._split_features_target(data_clean)
        
        # 3. División en conjuntos
        splits = self._split_data(X, y)
        
        # 4. Escalado
        scaled_splits = self._scale_features(splits)
        
        self.is_fitted = True
        return scaled_splits
    
    def _clean_data(self, df):
        """Limpia y prepara los datos"""
        data = df.copy()
        
        # Manejar valores cero (considerados como missing)
        for col in PREPROCESSING_CONFIG['columns_to_clean']:
            data[col] = data[col].replace(0, np.nan)
            # Imputar con mediana agrupada por Outcome - CORREGIDO
            data[col] = data.groupby(TARGET_NAME)[col].transform(
                lambda x: x.fillna(x.median())
            )
        
        print("✅ Datos limpiados exitosamente")
        return data
    
    def _split_features_target(self, df):
        """Separa características y variable objetivo"""
        X = df.drop(TARGET_NAME, axis=1)  # CORREGIDO
        y = df[TARGET_NAME]  # CORREGIDO
        return X, y

    def _split_data(self, X, y):
        """Divide en entrenamiento, validación y prueba"""
        test_size = PREPROCESSING_CONFIG['test_size']
        val_size = PREPROCESSING_CONFIG['val_size']
        random_state = PREPROCESSING_CONFIG['random_state']
        
        # Primera división
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, 
            test_size=test_size + val_size,
            random_state=random_state,
            stratify=y
        )
        
        # Segunda división
        test_ratio = test_size / (test_size + val_size)
        X_test, X_val, y_test, y_val = train_test_split(
            X_temp, y_temp,
            test_size=test_ratio,
            random_state=random_state,
            stratify=y_temp
        )
        
        splits = {
            'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
            'feature_names': X.columns.tolist()
        }
        
        print("✅ Datos divididos en entrenamiento, validación y prueba")
        return splits
    
    def _scale_features(self, splits):
        """Estandariza las características"""
        # Escalar datos de entrenamiento
        X_train_scaled = self.scaler.fit_transform(splits['X_train'])
        X_val_scaled = self.scaler.transform(splits['X_val'])
        X_test_scaled = self.scaler.transform(splits['X_test'])
        
        # Actualizar splits con datos escalados
        splits.update({
            'X_train_scaled': X_train_scaled,
            'X_val_scaled': X_val_scaled,
            'X_test_scaled': X_test_scaled
        })
        
        # Guardar scaler
        os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
        joblib.dump(self.scaler, f"{MODEL_SAVE_PATH}scaler.pkl")
        
        print("✅ Características escaladas y scaler guardado")
        return splits
    
    def transform_new_data(self, X):
        """Transforma nuevos datos usando el scaler ajustado"""
        if not self.is_fitted:
            raise ValueError("El preprocesador no ha sido ajustado. Llama a preprocess() primero.")
        
        return self.scaler.transform(X)