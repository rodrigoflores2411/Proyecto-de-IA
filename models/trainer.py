from sklearn.ensemble import RandomForestClassifier  
from sklearn.neural_network import MLPClassifier 
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score 
import joblib
import pandas as pd
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config import MODEL_PARAMS, MODEL_SAVE_PATH
class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0
    
    def initialize_models(self):
        """Inicializa los modelos con sus parámetros"""
        self.models = {
            'Random Forest': RandomForestClassifier(**MODEL_PARAMS['random_forest']),
            'XGBoost': XGBClassifier(**MODEL_PARAMS['xgboost']),
            'Neural Network': MLPClassifier(**MODEL_PARAMS['neural_network'])
        }
        print("✅ Modelos inicializados")
    
    def train_models(self, splits):
        """
        Entrena todos los modelos
        
        Args:
            splits (dict): Diccionario con datos divididos
            
        Returns:
            dict: Resultados del entrenamiento
        """
        self.results = {}
        
        for name, model in self.models.items():
            print(f"\n--- Entrenando {name} ---")
            
            # Seleccionar datos según el modelo
            if name == 'Neural Network':
                X_train = splits['X_train_scaled']
                X_val = splits['X_val_scaled']
            else:
                X_train = splits['X_train']
                X_val = splits['X_val']
            
            y_train = splits['y_train']
            y_val = splits['y_val']
            
            # Entrenamiento
            model.fit(X_train, y_train)
            
            # Evaluación
            model_results = self._evaluate_model(model, X_val, y_val, name)
            self.results[name] = model_results
            
            # Verificar si es el mejor modelo
            if model_results['auc'] > self.best_score:
                self.best_score = model_results['auc']
                self.best_model = model
                self.best_model_name = name
        
        print(f"\n🎯 Mejor modelo: {self.best_model_name} (AUC: {self.best_score:.4f})")
        return self.results
    
    def _evaluate_model(self, model, X_val, y_val, model_name):
        """Evalúa un modelo individual"""
        # Predicciones
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        # Métricas
        metrics = {
            'model': model,
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred),
            'recall': recall_score(y_val, y_pred),
            'f1_score': f1_score(y_val, y_pred),
            'auc': roc_auc_score(y_val, y_pred_proba),
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"✅ {model_name} - AUC: {metrics['auc']:.4f}, F1: {metrics['f1_score']:.4f}")
        return metrics
    
    def save_models(self):
        """Guarda todos los modelos entrenados"""
        os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
        
        for name, result in self.results.items():
            filename = f"{MODEL_SAVE_PATH}{name.lower().replace(' ', '_')}.pkl"
            joblib.dump(result['model'], filename)
            print(f"💾 Modelo guardado: {filename}")
        
        # Guardar el mejor modelo por separado
        if self.best_model:
            joblib.dump(self.best_model, f"{MODEL_SAVE_PATH}best_model.pkl")
            print(f"🏆 Mejor modelo guardado: {MODEL_SAVE_PATH}best_model.pkl")
    
    def get_training_summary(self):
        """Retorna un resumen del entrenamiento"""
        if not self.results:
            return None
        
        summary_data = []
        for name, result in self.results.items():
            summary_data.append({
                'Modelo': name,
                'Accuracy': f"{result['accuracy']:.4f}",
                'Precision': f"{result['precision']:.4f}",
                'Recall': f"{result['recall']:.4f}",
                'F1-Score': f"{result['f1_score']:.4f}",
                'AUC': f"{result['auc']:.4f}"
            })
        
        return pd.DataFrame(summary_data)
