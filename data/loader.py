# data/loader.py - Actualizado con Kaggle Hub

import pandas as pd
import numpy as np
import os
import sys
import kagglehub  # Nueva importación

# Añadir path para importar config
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config import COLUMN_NAMES, DATASET_SOURCES, TARGET_NAME

class DataLoader:
    def __init__(self):
        self.df = None
        self.dataset_path = None
    
    def load_data(self, source='kaggle'):
        """
        Carga datos desde Kaggle Hub, URL o archivo local
        
        Args:
            source (str): 'kaggle', 'url', o 'local'
        
        Returns:
            pandas.DataFrame: Dataset cargado
        """
        try:
            if source == 'kaggle':
                self.df = self._load_from_kaggle()
                print("✅ Datos cargados desde Kaggle Hub")
            elif source == 'url':
                self.df = pd.read_csv(DATASET_SOURCES['url'], names=COLUMN_NAMES)
                print("✅ Datos cargados desde URL")
            elif source == 'local':
                local_path = DATASET_SOURCES['local']
                if os.path.exists(local_path):
                    self.df = pd.read_csv(local_path)
                    # Verificar si necesita nombres de columnas
                    if len(self.df.columns) == len(COLUMN_NAMES):
                        self.df.columns = COLUMN_NAMES
                    print("✅ Datos cargados desde archivo local")
                else:
                    raise FileNotFoundError(f"Archivo no encontrado: {local_path}")
            else:
                raise ValueError("Fuente no válida. Use 'kaggle', 'url' o 'local'")
            
            self._validate_data()
            return self.df
            
        except Exception as e:
            print(f"❌ Error cargando datos: {e}")
            # Fallback a URL si Kaggle falla
            if source == 'kaggle':
                print("⚠️  Fallando a fuente URL...")
                return self.load_data(source='url')
            return None
    
    def _load_from_kaggle(self):
        """Carga datos desde Kaggle Hub"""
        try:
            # Descargar dataset
            dataset_name = DATASET_SOURCES['kaggle']['dataset_name']
            print(f"📥 Descargando dataset de Kaggle: {dataset_name}")
            
            path = kagglehub.dataset_download(dataset_name)
            self.dataset_path = path
            print(f"✅ Dataset descargado en: {path}")
            
            # Buscar archivo CSV en el directorio descargado
            csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
            if not csv_files:
                raise FileNotFoundError("No se encontraron archivos CSV en el dataset")
            
            # Usar el archivo específico si está configurado, sino el primer CSV
            expected_file = DATASET_SOURCES['kaggle'].get('file_name')
            if expected_file and expected_file in csv_files:
                csv_path = os.path.join(path, expected_file)
            else:
                csv_path = os.path.join(path, csv_files[0])
            
            print(f"📖 Leyendo archivo: {csv_path}")
            
            # Leer el archivo CSV
            df = pd.read_csv(csv_path)
            
            # Verificar y ajustar nombres de columnas si es necesario
            if len(df.columns) == len(COLUMN_NAMES):
                # Si coincide el número de columnas, usar nuestros nombres
                df.columns = COLUMN_NAMES
            elif 'Outcome' in df.columns or 'outcome' in df.columns:
                # Si ya tiene la columna objetivo, mantener los nombres originales
                print("✅ Usando nombres de columnas del dataset original")
            else:
                # Si no coincide, intentar inferir
                print("⚠️  Los nombres de columnas no coinciden, intentando inferir...")
                if len(df.columns) >= len(COLUMN_NAMES):
                    df = df.iloc[:, :len(COLUMN_NAMES)]
                    df.columns = COLUMN_NAMES
            
            return df
            
        except Exception as e:
            print(f"❌ Error cargando desde Kaggle: {e}")
            raise
    
    def _validate_data(self):
        """Valida la estructura y calidad básica de los datos"""
        if self.df is None:
            raise ValueError("No hay datos cargados")
        
        # Verificar que tenga las columnas necesarias
        required_columns = [TARGET_NAME] + [col for col in COLUMN_NAMES if col != TARGET_NAME]
        
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            print(f"⚠️  Columnas faltantes: {missing_columns}")
            print(f"📊 Columnas disponibles: {self.df.columns.tolist()}")
            
            # Intentar mapear columnas similares
            column_mapping = {}
            for required in missing_columns:
                for actual in self.df.columns:
                    if required.lower() in actual.lower() or actual.lower() in required.lower():
                        column_mapping[actual] = required
                        break
            
            if column_mapping:
                self.df = self.df.rename(columns=column_mapping)
                print(f"✅ Columnas mapeadas: {column_mapping}")
        
        # Verificar variable objetivo
        if TARGET_NAME not in self.df.columns:
            # Buscar columnas que puedan ser la variable objetivo
            possible_targets = [col for col in self.df.columns if 'outcome' in col.lower() or 'target' in col.lower() or 'diabetes' in col.lower()]
            if possible_targets:
                # Usar la primera columna que parece ser la objetivo
                actual_target = possible_targets[0]
                self.df = self.df.rename(columns={actual_target: TARGET_NAME})
                print(f"✅ Variable objetivo identificada: {actual_target} -> {TARGET_NAME}")
            else:
                # Si no se encuentra, usar la última columna
                last_column = self.df.columns[-1]
                self.df = self.df.rename(columns={last_column: TARGET_NAME})
                print(f"⚠️  Usando última columna como objetivo: {last_column} -> {TARGET_NAME}")
        
        print(f"📊 Dataset validado: {self.df.shape[0]} filas, {self.df.shape[1]} columnas")
        print(f"🎯 Distribución de objetivo: {self.df[TARGET_NAME].value_counts().to_dict()}")
    
    def get_data_info(self):
        """Retorna información básica del dataset"""
        if self.df is None:
            return None
        
        info = {
            'shape': self.df.shape,
            'columns': self.df.columns.tolist(),
            'target_distribution': self.df[TARGET_NAME].value_counts().to_dict(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'data_types': self.df.dtypes.to_dict(),
            'source_path': self.dataset_path if self.dataset_path else 'URL/Local'
        }
        
        return info
    
    def save_local_copy(self, path=None):
        """Guarda una copia local del dataset"""
        if path is None:
            path = DATASET_SOURCES['local']
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.df.to_csv(path, index=False)
        print(f"💾 Copia local guardada en: {path}")
        return path