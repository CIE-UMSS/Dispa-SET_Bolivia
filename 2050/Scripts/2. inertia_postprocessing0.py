# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 10:04:23 2024

@author: navia
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dispaset as ds
import seaborn as sns

# Diccionario de escenarios: nombre -> ruta
escenarios = {
    'EPI': '../Simulations/BOLIVIA_2050_EPI_FFR',
    'EPI_SI': '../Simulations/BOLIVIA_2050_EPI_SI_FFR',    
    'EPI_DI': '../Simulations/BOLIVIA_2050_EPI_DI_FFR', 
    'NZE': '../Simulations/BOLIVIA_2050_NZE_FFR',
    'NZE_SI': '../Simulations/BOLIVIA_2050_NZE_SI_FFR',
    'NZE_DI': '../Simulations/BOLIVIA_2050_NZE_DI_FFR' 
}

df_all = []

def get_season(date):
    return 'Seco' if 5 <= date.month <= 10 else 'Lluvioso'

# Cargar datos
for nombre, ruta in escenarios.items():
    inputs, results = ds.get_sim_results(path=ruta, cache=False)
    df = results['OutputSysInertia'].copy().to_frame(name='OutputSysInertia')

    # Asegurar que el índice es datetime
    df.index = pd.to_datetime(df.index)

    # Convertir el índice a serie para trabajar con .dt
    index_series = df.index.to_series()

    df['Fecha'] = index_series.dt.date
    df['Hora'] = index_series.dt.hour
    df['Periodo'] = index_series.map(get_season)
    df['Escenario'] = nombre

    # Renombrar la columna '0' a 'SystemInertia'
    df = df.rename(columns={df.columns[0]: 'SystemInertia'})
    
    df_all.append(df)

# Concatenar todo
df_concat = pd.concat(df_all).reset_index()

# Estilo de gráficos
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.1)

colores_escenarios = {
    'EPI': '#F4A261',
    'EPI_SI': '#2A9D8F',    
    'EPI_DI': '#264653', 
    'NZE': '#E76F51',
    'NZE_SI': '#457B9D',
    'NZE_DI': '#A8DADC' 
}

# Ahora df_concat ya tiene columnas: ['index', 'SystemInertia', 'Fecha', 'Hora', 'Periodo', 'Escenario']

#%%
# === 1. SystemInertia total anual por zona y escenario ===

total = df_concat.groupby('Escenario')['SystemInertia'].sum().reset_index()

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(total['Escenario'], total['SystemInertia'],
              color=[colores_escenarios[esc] for esc in total['Escenario']],
              edgecolor='black', alpha=0.9)

# Añadir valores sobre las barras
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.02 * total['SystemInertia'].max(),
            f'{height:,.0f}', ha='center', va='bottom', fontsize=9)

ax.set_xlabel('Escenario', fontsize=12)
ax.set_ylabel('SystemInertia total (GWs)', fontsize=12)
ax.set_title('SystemInertia total anual por escenario', fontsize=14, fontweight='bold')
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.show()



#%%

# === 2. SystemInertia horario por zona y escenario (Boxplot) ===
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(data=df_concat, x='Escenario', y='SystemInertia',
            palette=colores_escenarios, fliersize=2, linewidth=1, ax=ax)

ax.set_title('Distribución horaria del SystemInertia por escenario', fontsize=14, fontweight='bold')
ax.set_ylabel('SystemInertia horario (GWs)', fontsize=12)
ax.set_xlabel('Escenario', fontsize=12)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.show()



#%%
# === 3. Inertia promedio diario por zona (Serie Temporal) ===
# Asegurarse que 'index' sea datetime
df_concat['index'] = pd.to_datetime(df_concat['index'])

# Extraer solo la fecha para el promedio diario
df_concat['Fecha'] = df_concat['index'].dt.date

# Agrupar por escenario y fecha para obtener promedio diario
df_diario = df_concat.groupby(['Escenario', 'Fecha'])['SystemInertia'].mean().reset_index()

# Escenarios únicos
escenarios = df_diario['Escenario'].unique()

# Configurar subplots
n = len(escenarios)
cols = 3  # ajustar según número de escenarios
rows = (n + cols - 1) // cols

fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), sharex=True, sharey=True)
axs = axs.flatten()

# Graficar cada escenario en su subplot
for i, esc in enumerate(escenarios):
    ax = axs[i]
    df_esc = df_diario[df_diario['Escenario'] == esc]
    ax.plot(df_esc['Fecha'], df_esc['SystemInertia'], color='steelblue', linewidth=1.6)
    ax.set_title(f'Escenario: {esc}', fontsize=13, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    if i % cols == 0:
        ax.set_ylabel('SystemInertia diario (MWh)', fontsize=11)
    if i // cols == rows - 1:
        ax.set_xlabel('Fecha', fontsize=11)

# Eliminar subplots vacíos
for j in range(i + 1, len(axs)):
    fig.delaxes(axs[j])

# Título general
fig.suptitle('SystemInertia promedio diario por escenario', fontsize=15, fontweight='bold', y=1.03)
plt.tight_layout()
plt.show()


#%%
# === 3.1 Inertia promedio diario por zona (Serie Temporal) ===
# Asegurarse que 'index' sea datetime

import matplotlib.pyplot as plt
import pandas as pd

# Asegurarse que 'index' sea datetime
df_concat['index'] = pd.to_datetime(df_concat['index'])

# Extraer solo la fecha para el promedio diario
df_concat['Fecha'] = df_concat['index'].dt.date

# Agrupar por escenario y fecha para obtener promedio, mínimo y máximo diario
df_stats = df_concat.groupby(['Escenario', 'Fecha'])['SystemInertia'].agg(['mean', 'min', 'max']).reset_index()

# Escenarios únicos
escenarios = df_stats['Escenario'].unique()

# Configurar subplots
n = len(escenarios)
cols = 3  # ajustar según número de escenarios
rows = (n + cols - 1) // cols

fig, axs = plt.subplots(rows, cols, figsize=(5.8 * cols, 4.2 * rows), sharex=True, sharey=True)
axs = axs.flatten()

# Colores
line_color = 'steelblue'
fill_color = 'lightsteelblue'

# Graficar cada escenario
for i, esc in enumerate(escenarios):
    ax = axs[i]
    df_esc = df_stats[df_stats['Escenario'] == esc]

    # Relleno de min a max (banda de incertidumbre)
    ax.fill_between(df_esc['Fecha'], df_esc['min'], df_esc['max'], color=fill_color, alpha=0.4)

    # Línea del promedio
    ax.plot(df_esc['Fecha'], df_esc['mean'], color=line_color, linewidth=1.8)

    ax.set_title(f'Escenario: {esc}', fontsize=13, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    if i % cols == 0:
        ax.set_ylabel('SystemInertia diario (MW·s)', fontsize=11)
    if i // cols == rows - 1:
        ax.set_xlabel('Fecha', fontsize=11)
    ax.tick_params(labelsize=9)

# Eliminar subplots vacíos
for j in range(i + 1, len(axs)):
    fig.delaxes(axs[j])

# Título general
fig.suptitle('SystemInertia diario promedio con rango (min–max) por escenario', 
             fontsize=16, fontweight='bold', y=1.03)

plt.tight_layout()
plt.subplots_adjust(top=0.93)
plt.show()




#%%
# === 4. Mapa de calor: SystemInertia medio por zona y hora ===
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Asegurar que el índice sea datetime y extraer la hora si no está
df_concat['index'] = pd.to_datetime(df_concat['index'])
df_concat['Hora'] = df_concat['index'].dt.hour

# Obtener escenarios y configurar grilla
escenarios = df_concat['Escenario'].unique()
n = len(escenarios)
cols = 3
rows = (n + cols - 1) // cols

# Crear figura
fig = plt.figure(figsize=(8 * cols, 4 * rows))
gs = gridspec.GridSpec(rows, cols + 1, width_ratios=[1]*cols + [0.05], hspace=0.25, wspace=0.15)

# Subplots y colorbar
axs = [fig.add_subplot(gs[i // cols, i % cols]) for i in range(n)]
cbar_ax = fig.add_subplot(gs[:, -1])  # eje para colorbar

# Escala común
vmin = df_concat['SystemInertia'].mean() - 2 * df_concat['SystemInertia'].std()
vmax = df_concat['SystemInertia'].mean() + 2 * df_concat['SystemInertia'].std()

# Dibujar heatmaps
for i, esc in enumerate(escenarios):
    ax = axs[i]
    df_esc = df_concat[df_concat['Escenario'] == esc]
    df_heatmap = df_esc.groupby('Hora')['SystemInertia'].mean().to_frame().T  # Solo 1 fila

    sns.heatmap(df_heatmap, cmap='viridis', linewidths=0.4, annot=True, fmt=".0f",
                ax=ax, cbar=i == 0, cbar_ax=cbar_ax if i == 0 else None,
                vmin=vmin, vmax=vmax, annot_kws={'fontsize': 14})

    ax.set_title(f'Escenario: {esc}', fontsize=16, fontweight='bold')
    ax.set_xlabel('Hora del día', fontsize=14)
    ax.set_ylabel('SystemInertia', fontsize=14)
    ax.tick_params(labelsize=12)
    ax.set_yticklabels([''])  # Quitar etiquetas del eje Y ya que es solo una fila

# Ajustar colorbar
cbar_ax.tick_params(labelsize=12)
cbar_ax.set_ylabel('SystemInertia (MWh)', fontsize=16, labelpad=20)

# Título global
fig.suptitle('SystemInertia medio por hora del día – por escenario',
             fontsize=20, fontweight='bold', y=1.02)

plt.tight_layout()
plt.subplots_adjust(right=0.92, top=0.93)
plt.show()
