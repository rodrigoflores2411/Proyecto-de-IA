import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import TARGET_NAME, PLOT_SAVE_PATH
import os

class DataExplorer:
    def __init__(self, df):
        self.df = df
        self.setup_plot_style()
    
    def setup_plot_style(self):
        """Configura el estilo de los gráficos"""
        plt.style.use('default')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (10, 6)
    
    def generate_summary_report(self):
        """Genera un reporte completo de análisis exploratorio"""
        report = {}
        
        report['basic_info'] = self.get_basic_info()
        report['statistical_summary'] = self.get_statistical_summary()
        report['correlation_analysis'] = self.get_correlation_analysis()
        
        # Generar gráficos
        self.plot_target_distribution()
        self.plot_feature_distributions()
        self.plot_correlation_heatmap()
        self.plot_feature_vs_target()
        
        return report
    
    def get_basic_info(self):
        """Retorna información básica del dataset"""
        info = {
            'shape': self.df.shape,
            'data_types': self.df.dtypes.to_dict(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'duplicate_rows': self.df.duplicated().sum()
        }
        return info
    
    def get_statistical_summary(self):
        """Retorna estadísticas descriptivas"""
        return self.df.describe().to_dict()
    
    def get_correlation_analysis(self):
        """Analiza correlaciones"""
        corr_matrix = self.df.corr()
        target_correlations = corr_matrix[TARGET_NAME].sort_values(ascending=False)
        
        return {
            'correlation_matrix': corr_matrix.to_dict(),
            'target_correlations': target_correlations.to_dict()
        }
    
    def plot_target_distribution(self):
        """Grafica la distribución de la variable objetivo"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Gráfico de pie
        target_counts = self.df[TARGET_NAME].value_counts()
        axes[0].pie(target_counts.values, labels=['No Diabetes', 'Diabetes'], 
                   autopct='%1.1f%%', startangle=90)
        axes[0].set_title('Distribución de Diabetes')
        
        # Gráfico de barras
        sns.countplot(data=self.df, x=TARGET_NAME, ax=axes[1])
        axes[1].set_title('Distribución de Resultados')
        axes[1].set_xticklabels(['No Diabetes', 'Diabetes'])
        
        plt.tight_layout()
        self._save_plot('target_distribution.png')
        plt.show()
    
    def plot_feature_distributions(self):
        """Grafica distribuciones de todas las características"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        n_cols = 3
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
        axes = axes.ravel()
        
        for i, col in enumerate(numeric_cols):
            self.df[col].hist(bins=30, ax=axes[i], alpha=0.7)
            axes[i].set_title(f'Distribución de {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frecuencia')
        
        # Ocultar ejes vacíos
        for i in range(len(numeric_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        self._save_plot('feature_distributions.png')
        plt.show()
    
    def plot_correlation_heatmap(self):
        """Grafica matriz de correlación"""
        plt.figure(figsize=(10, 8))
        corr_matrix = self.df.corr()
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, mask=mask, fmt='.2f')
        plt.title('Matriz de Correlación')
        
        self._save_plot('correlation_heatmap.png')
        plt.show()
    
    def plot_feature_vs_target(self):
        """Grafica relación entre características y objetivo"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        numeric_cols = numeric_cols[numeric_cols != TARGET_NAME]
        
        n_cols = 3
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
        axes = axes.ravel()
        
        for i, col in enumerate(numeric_cols):
            self.df.boxplot(column=col, by=TARGET_NAME, ax=axes[i])
            axes[i].set_title(f'{col} vs Diabetes')
            axes[i].set_xlabel('Diabetes')
        
        plt.suptitle('')  # Eliminar título automático
        plt.tight_layout()
        self._save_plot('feature_vs_target.png')
        plt.show()
    
    def _save_plot(self, filename):
        """Guarda el gráfico en la carpeta de resultados"""
        os.makedirs(PLOT_SAVE_PATH, exist_ok=True)
        plt.savefig(f"{PLOT_SAVE_PATH}{filename}", dpi=300, bbox_inches='tight')
        print(f"📊 Gráfico guardado: {PLOT_SAVE_PATH}{filename}")
