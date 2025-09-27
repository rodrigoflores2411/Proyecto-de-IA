from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix,classification_report, roc_curve)  # IMPORT CORREGIDO
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import sys

# Añadir path para importar config
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config import PLOT_SAVE_PATH, TARGET_NAME

class ModelEvaluator:
    def __init__(self, models_results, splits):
        self.models_results = models_results
        self.splits = splits
        self.test_results = {}
        for name, result in self.models_results.items():
            print(f"\n--- Evaluando {name} en Test ---")
            
            # Seleccionar datos según el modelo
            if name == 'Neural Network':
                X_test = self.splits['X_test_scaled']
            else:
                X_test = self.splits['X_test']
            
            y_test = self.splits['y_test']
            model = result['model']
            
            # Predicciones
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Métricas
            test_metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'auc': roc_auc_score(y_test, y_pred_proba),
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            self.test_results[name] = test_metrics
            print(f"✅ {name} - Test AUC: {test_metrics['auc']:.4f}")
        
        return self.test_results
    
    def generate_comparison_report(self):
        """Genera un reporte comparativo de todos los modelos"""
        if not self.test_results:
            self.evaluate_on_test()
        
        # Crear DataFrame comparativo
        comparison_data = []
        for name, metrics in self.test_results.items():
            comparison_data.append({
                'Modelo': name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score'],
                'AUC-ROC': metrics['auc']
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        return df_comparison
    
    def plot_confusion_matrices(self):
        """Grafica matrices de confusión para todos los modelos"""
        n_models = len(self.test_results)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        
        if n_models == 1:
            axes = [axes]
        
        for i, (name, metrics) in enumerate(self.test_results.items()):
            cm = confusion_matrix(self.splits['y_test'], metrics['predictions'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_title(f'{name}\nAccuracy: {metrics["accuracy"]:.3f}')
            axes[i].set_xlabel('Predicho')
            axes[i].set_ylabel('Real')
            axes[i].set_xticklabels(['No Diabetes', 'Diabetes'])
            axes[i].set_yticklabels(['No Diabetes', 'Diabetes'])
        
        plt.tight_layout()
        self._save_plot('confusion_matrices.png')
        plt.show()
    
    def plot_roc_curves(self):
        """Grafica curvas ROC para todos los modelos"""
        plt.figure(figsize=(10, 8))
        
        for name, metrics in self.test_results.items():
            fpr, tpr, _ = roc_curve(self.splits['y_test'], metrics['probabilities'])
            auc_score = metrics['auc']
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Clasificador aleatorio')
        plt.xlabel('Tasa de Falsos Positivos')
        plt.ylabel('Tasa de Verdaderos Positivos')
        plt.title('Curvas ROC - Comparación de Modelos')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        self._save_plot('roc_curves.png')
        plt.show()
    
    def plot_metrics_comparison(self):
        """Grafica comparación de métricas entre modelos"""
        comparison_df = self.generate_comparison_report()
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            bars = axes[i].bar(comparison_df['Modelo'], comparison_df[metric])
            axes[i].set_title(f'Comparación de {metric}')
            axes[i].set_ylabel(metric)
            axes[i].tick_params(axis='x', rotation=45)
            
            # Añadir valores en las barras
            for bar, value in zip(bars, comparison_df[metric]):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        # Ocultar el último subplot si es necesario
        if len(metrics) < len(axes):
            for i in range(len(metrics), len(axes)):
                axes[i].set_visible(False)
        
        plt.tight_layout()
        self._save_plot('metrics_comparison.png')
        plt.show()
    
    def _save_plot(self, filename):
        """Guarda el gráfico"""
        os.makedirs(PLOT_SAVE_PATH, exist_ok=True)
        plt.savefig(f"{PLOT_SAVE_PATH}{filename}", dpi=300, bbox_inches='tight')