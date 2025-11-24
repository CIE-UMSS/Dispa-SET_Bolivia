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


zonas = ['CE', 'NO', 'OR', 'SU']
df_all = []

def get_season(date):
    return 'Seco' if 5 <= date.month <= 10 else 'Lluvioso'

# Cargar datos
for nombre, ruta in escenarios.items():
    inputs, results = ds.get_sim_results(path=ruta, cache=False)
    df = results['OutputCurtailedPower'].copy()
    df['Escenario'] = nombre
    df['Periodo'] = df.index.map(get_season)
    df_all.append(df)


# Concatenar
df_concat = pd.concat(df_all)
df_concat = df_concat.reset_index()

# Asegurarse que el índice sea datetime si no lo es aún
df_concat['index'] = pd.to_datetime(df_concat['index'])

# Separar fecha y hora en nuevas columnas
df_concat['Fecha'] = df_concat['index'].dt.date
df_concat['Hora'] = df_concat['index'].dt.hour
df_melted = df_concat.melt(id_vars=['Escenario', 'Periodo','Fecha','Hora','index'], value_vars=zonas,
                            var_name='Zona', value_name='Curtailment')

# Estilo general
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.1)
colors_zones = {
    'CE': '#0099cc',
    'OR': '#ff9966',
    'NO': '#66cc66',
    'SU': '#800080'
}
colores_escenarios = {
    'EPI': '#F4A261',
    'EPI_SI': '#2A9D8F',    
    'EPI_DI': '#264653', 
    'NZE': '#E76F51',
    'NZE_SI': '#457B9D',
    'NZE_DI': '#A8DADC' 
}
#%%
# === 1. Curtailment total anual por zona y escenario ===
total = df_melted.groupby(['Escenario', 'Zona'])['Curtailment'].sum().reset_index()
zonas = total['Zona'].unique()
escenarios_list = total['Escenario'].unique()
# Primero define width dependiendo del número de escenarios
width = 0.8 / len(escenarios_list)

# Ahora ajusta x con suficiente espacio entre grupos de zonas
x = np.arange(len(zonas)) * (width * len(escenarios_list) + 0.1)

fig, ax = plt.subplots(figsize=(20, 6))
for i, esc in enumerate(escenarios_list):
    data_esc = total[total['Escenario'] == esc]['Curtailment'].values
    bars = ax.bar(x + (i - len(escenarios_list)/2)*width + width/2, data_esc, width,
                  label=esc, color=colores_escenarios[esc], edgecolor='black', alpha=0.9)
    for bar, val in zip(bars, data_esc):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02 * max(data_esc),
                f'{val:,.0f}', ha='center', va='bottom', fontsize=9)

ax.set_xlabel('Zona', fontsize=12)
ax.set_ylabel('Curtailment total (MWh)', fontsize=12)
ax.set_title('Curtailment total anual por zona y escenario', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(zonas, fontsize=11)

# Eliminar las líneas de grid
ax.grid(False)  # Esto elimina las líneas de rejilla (grid)

# Separation lines between technologies
for i in range(len(zonas) - 1):
    ax.axvline(x=i + 0.5, color='gray', linestyle='--', linewidth=1)

ax.legend(title='Escenario', fontsize=10)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.show()


#%%

# === 2. Curtailment horario por zona y escenario (Boxplot) ===
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(data=df_melted, x='Zona', y='Curtailment', hue='Escenario',
            palette=colores_escenarios, fliersize=2, linewidth=1, ax=ax)
ax.set_title('Distribución del Curtailment horario por zona y escenario', fontsize=14, fontweight='bold')
ax.set_ylabel('Curtailment horario (MWh)', fontsize=12)
ax.set_xlabel('Zona', fontsize=12)
ax.legend(title='Escenario', fontsize=10)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.show()
#%%

# === 2.1 Curtailment horario por zona y escenario (Boxplot) analisis de temporadas===

import seaborn as sns
import matplotlib.pyplot as plt

# Crear subplots para 'Lluvioso' y 'Seco'
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Filtrar los datos por cada periodo
df_lluvioso = df_melted[df_melted['Periodo'] == 'Lluvioso']
df_seco = df_melted[df_melted['Periodo'] == 'Seco']

# Boxplot para 'Lluvioso'
sns.boxplot(data=df_lluvioso, x='Zona', y='Curtailment', hue='Escenario', 
            palette=colores_escenarios, fliersize=2, linewidth=1, ax=ax1)
ax1.set_title('Distribución del Curtailment horario - Periodo Lluvioso', fontsize=14, fontweight='bold')
ax1.set_ylabel('Curtailment horario (MWh)', fontsize=12)
ax1.set_xlabel('Zona', fontsize=12)
ax1.legend(title='Escenario', fontsize=10)
ax1.spines[['top', 'right']].set_visible(False)

# Boxplot para 'Seco'
sns.boxplot(data=df_seco, x='Zona', y='Curtailment', hue='Escenario', 
            palette=colores_escenarios, fliersize=2, linewidth=1, ax=ax2)
ax2.set_title('Distribución del Curtailment horario - Periodo Seco', fontsize=14, fontweight='bold')
ax2.set_ylabel('Curtailment horario (MWh)', fontsize=12)
ax2.set_xlabel('Zona', fontsize=12)
ax2.legend(title='Escenario', fontsize=10)
ax2.spines[['top', 'right']].set_visible(False)

# Ajustar la disposición
plt.tight_layout()
plt.show()



#%%
# === 2.2 Curtailment horario por zona y escenario (Boxplot) analisis de temporadas===
# violin + swarmplot
import seaborn as sns
import matplotlib.pyplot as plt

# Crear subplots para 'Lluvioso' y 'Seco'
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Filtrar los datos por cada periodo
df_lluvioso = df_melted[df_melted['Periodo'] == 'Lluvioso']
df_seco = df_melted[df_melted['Periodo'] == 'Seco']

# Lluvioso
sns.violinplot(data=df_lluvioso, x='Zona', y='Curtailment', hue='Escenario',
               palette=colores_escenarios, ax=ax1, cut=0, inner=None)
sns.swarmplot(data=df_lluvioso, x='Zona', y='Curtailment', hue='Escenario',
              dodge=True, palette=['k']*len(colores_escenarios), alpha=0.3, size=2, ax=ax1)
ax1.set_title('Distribución del Curtailment horario - Periodo Lluvioso', fontsize=14, fontweight='bold')
ax1.set_ylabel('Curtailment horario (MWh)', fontsize=12)
ax1.set_xlabel('Zona', fontsize=12)
ax1.spines[['top', 'right']].set_visible(False)
ax1.legend_.remove()  # remover leyenda duplicada del swarmplot

# Seco
sns.violinplot(data=df_seco, x='Zona', y='Curtailment', hue='Escenario',
               palette=colores_escenarios, ax=ax2, cut=0, inner=None)
sns.swarmplot(data=df_seco, x='Zona', y='Curtailment', hue='Escenario',
              dodge=True, palette=['k']*len(colores_escenarios), alpha=0.3, size=2, ax=ax2)
ax2.set_title('Distribución del Curtailment horario - Periodo Seco', fontsize=14, fontweight='bold')
ax2.set_ylabel('Curtailment horario (MWh)', fontsize=12)
ax2.set_xlabel('Zona', fontsize=12)
ax2.spines[['top', 'right']].set_visible(False)

# Leyenda única
handles, labels = ax2.get_legend_handles_labels()
fig.legend(handles[:len(colores_escenarios)], labels[:len(colores_escenarios)],
           title='Escenario', loc='upper center', ncol=len(colores_escenarios), fontsize=10)

plt.tight_layout()
plt.show()




#%%
# === 3. Curtailment promedio diario por zona (Serie Temporal) ===
# Asegúrate de que la columna 'Fecha' sea datetime
df_melted['Fecha'] = pd.to_datetime(df_melted['Fecha'])

# Obtener escenarios y zonas
escenarios = df_melted['Escenario'].unique()
zonas = df_melted['Zona'].unique()

# Grupos diarios
df_diario = df_melted.groupby(['Escenario', 'Fecha', 'Zona'])['Curtailment'].mean().reset_index()

# Configuración de subplots (en una grilla)
n = len(escenarios)
cols = 3  # puedes ajustar esto si tienes muchos escenarios
rows = (n + cols - 1) // cols

fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), sharex=True, sharey=True)
axs = axs.flatten()

# Colores para las zonas
colors_zones = {
    'CE': '#0099cc',
    'OR': '#ff9966',
    'NO': '#66cc66',
    'SU': '#800080'
}

# Graficar cada escenario en su subplot
for i, esc in enumerate(escenarios):
    ax = axs[i]
    df_esc = df_diario[df_diario['Escenario'] == esc]
    for zona in zonas:
        data_zona = df_esc[df_esc['Zona'] == zona]
        ax.plot(data_zona['Fecha'], data_zona['Curtailment'],
                label=zona, color=colors_zones.get(zona, 'gray'), linewidth=1.6)
    
    ax.set_title(f'Escenario: {esc}', fontsize=13, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    if i % cols == 0:
        ax.set_ylabel('Curtailment diario (MWh)', fontsize=11)
    if i // cols == rows - 1:
        ax.set_xlabel('Fecha', fontsize=11)

# Quitar subplots vacíos si hay menos escenarios que subplots
for j in range(i + 1, len(axs)):
    fig.delaxes(axs[j])

# Leyenda global
fig.legend(zonas, title='Zona', loc='upper center', ncol=len(zonas), fontsize=10)
fig.suptitle('Curtailment promedio diario por zona y escenario', fontsize=15, fontweight='bold', y=1.03)
plt.tight_layout()
plt.show()








#%%
# === 4. Mapa de calor: Curtailment medio por zona y hora ===
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# df_melted['Hora'] = pd.to_datetime(df_melted['index']).dt.hour

# Escenarios y configuración de grilla
escenarios = df_melted['Escenario'].unique()
n = len(escenarios)
cols = 3
rows = (n + cols - 1) // cols

# Crear figura con más altura y espacio para colorbar
fig = plt.figure(figsize=(8 * cols, 5 * rows))  # Más altura
gs = gridspec.GridSpec(rows, cols + 1, width_ratios=[1]*cols + [0.05], hspace=0.25, wspace=0.15)

# Subplots y colorbar
axs = [fig.add_subplot(gs[i // cols, i % cols]) for i in range(n)]
cbar_ax = fig.add_subplot(gs[:, -1])  # eje único para la barra de color

# Escala global uniforme
vmin = df_melted['Curtailment'].mean() - 2 * df_melted['Curtailment'].std()
vmax = df_melted['Curtailment'].mean() + 2 * df_melted['Curtailment'].std()

# Dibujar heatmaps
for i, esc in enumerate(escenarios):
    ax = axs[i]
    df_esc = df_melted[df_melted['Escenario'] == esc]
    df_heatmap = df_esc.groupby(['Zona', 'Hora'])['Curtailment'].mean().unstack('Hora')

    sns.heatmap(df_heatmap, cmap='viridis', linewidths=0.4, annot=True, fmt=".0f",
                ax=ax, cbar=i == 0, cbar_ax=cbar_ax if i == 0 else None,
                vmin=vmin, vmax=vmax, annot_kws={'rotation': 90, 'fontsize': 16})

    ax.set_title(f'Escenario: {esc}', fontsize=18, fontweight='bold')
    ax.set_xlabel('Hora del día', fontsize=18)
    ax.set_ylabel('Zona', fontsize=18)
    ax.tick_params(labelsize=16)  # 👈 Cambiar tamaño de ticks

# Ajustar barra de color
cbar_ax.tick_params(labelsize=12)  # 👈 Tamaño de fuente en números de la colorbar
cbar_ax.set_ylabel('Curtailment (MWh)', fontsize=18, labelpad=20)  # 👈 Etiqueta colorbar


# Título general
fig.suptitle('Mapa de calor del Curtailment medio por zona y hora – por escenario',
             fontsize=20, fontweight='bold', y=1.02)

plt.tight_layout()
plt.subplots_adjust(right=0.92, top=0.93)
plt.show()
