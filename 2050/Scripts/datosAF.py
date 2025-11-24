# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 12:16:24 2025

@author: navia
"""

import pandas as pd

# Cargar el archivo CSV
# file_path = 'C:/Users/navia/Documents/DISPASET MODELS/10.1TESMEP/MIX CARLOS (EPI)/wind_cp_per_bus.csv'  # Cambia el nombre si tu archivo se llama distinto
file_path = 'C:/Users/navia/Documents/DISPASET MODELS/10.1TESMEP/MIX CARLOS (EPI)/solar_cp_per_bus.csv'  # Cambia el nombre si tu archivo se llama distinto

df = pd.read_csv(file_path, index_col=0)

# Inicializar diccionario para guardar resultados
stats = {}

for column in df.columns:
    data = df[column]
    stats[column] = {
        'Mean': data.mean(),
        'Median (P50)': data.median(),
        'Std. Deviation': data.std(),
        'P10': data.quantile(0.10),
        'P90': data.quantile(0.90),
        'Max': data.max(),
        'Min': data.min(),
        'Hours > 0.5': (data > 0.5).sum(),
        '% Hours > 0.5': 100 * (data > 0.5).mean()
    }

# Convertir a DataFrame para visualización/tablas
summary_df = pd.DataFrame(stats).T  # Transponer para tener zonas como filas

# Mostrar la tabla
print(summary_df.round(3))  # Redondear a 3 decimales

