# config.py - Actualizado con Kaggle Hub

# Configuración de fuentes de datos
DATASET_SOURCES = {
    'kaggle': {
        'dataset_name': 'wasiqaliyasir/diabates-dataset',
        'file_name': 'diabetes.csv'  # Asumiendo que este es el nombre del archivo en el dataset
    },
    'url': 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv',
    'local': 'data/diabetes.csv'
}

# Rutas de archivos
DATA_PATH = DATASET_SOURCES['url']  # Por defecto
LOCAL_DATA_PATH = DATASET_SOURCES['local']
MODEL_SAVE_PATH = "results/models/"
PLOT_SAVE_PATH = "results/plots/"

# Nombres de columnas
FEATURE_NAMES = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]
TARGET_NAME = 'Outcome'
COLUMN_NAMES = FEATURE_NAMES + [TARGET_NAME]

# Resto de la configuración se mantiene igual...
MODEL_PARAMS = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': 42
    },
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42
    },
    'neural_network': {
        'hidden_layer_sizes': (100, 50),
        'max_iter': 1000,
        'random_state': 42
    }
}

PREPROCESSING_CONFIG = {
    'columns_to_clean': ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI'],
    'test_size': 0.2,
    'val_size': 0.2,
    'random_state': 42
}

RISK_THRESHOLDS = {
    'low': 0.3,
    'medium': 0.7,
    'high': 1.0
}