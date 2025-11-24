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
    'EPI_NC': '../Simulations/BOLIVIA_2050_EPI_FFR',
    'EPI_SC': '../Simulations/BOLIVIA_2050_EPI_SI_FFR',    
    'EPI_DC': '../Simulations/BOLIVIA_2050_EPI_DI_FFR', 
    'NZE_NC': '../Simulations/BOLIVIA_2050_NZE_FFR',
    'NZE_SC': '../Simulations/BOLIVIA_2050_NZE_SI_FFR',
    'NZE_DC': '../Simulations/BOLIVIA_2050_NZE_DI_FFR' 
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
    'EPI_NC':  '#1f77b4',  # azul
    'EPI_SC':  '#5fa2d9',  # azul claro
    'EPI_DC':  '#12507d',  # azul oscuro

    'NZE_NC':  '#ff7f0e',  # naranja
    'NZE_SC':  '#ffb666',  # naranja claro
    'NZE_DC':  '#b25900',  # naranja oscuro

}

# Ahora df_concat ya tiene columnas: ['index', 'SystemInertia', 'Fecha', 'Hora', 'Periodo', 'Escenario']

#%%
# === 1. Annual total SystemInertia by scenario ===

# Generar orden deseado por escenario (S0 a S6) y enfoque (NC, DC, SC)
orden_escenarios = ['EPI_NC', 'EPI_DC', 'EPI_SC', 'NZE_NC', 'NZE_DC', 'NZE_SC']

# Agrupar y ordenar
total = df_concat.groupby('Escenario')['SystemInertia'].sum().reset_index()
total['Escenario'] = pd.Categorical(total['Escenario'], categories=orden_escenarios, ordered=True)
total = total.sort_values('Escenario')

# Graficar
fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(total['Escenario'], total['SystemInertia'],
              color=[colores_escenarios[esc] for esc in total['Escenario']],
              edgecolor='black', alpha=0.9)

# Añadir valores sobre las barras
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.02 * total['SystemInertia'].max(),
            f'{height:,.0f}', ha='center', va='bottom', fontsize=9)

# Estética del gráfico
ax.set_xlabel('Scenario', fontsize=12)
ax.set_ylabel('Total System Inertia (GWs)', fontsize=12)
ax.set_title('Annual Total System Inertia by Scenario', fontsize=14, fontweight='bold')
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.show()

#%%
# === 1.1 Annual total SystemInertia by scenario con curvas tendencia ===
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

# === Preparar datos ===
orden_escenarios = ['EPI_NC', 'EPI_DC', 'EPI_SC', 'NZE_NC', 'NZE_DC', 'NZE_SC']
total = df_concat.groupby('Escenario')['SystemInertia'].sum().reset_index()
total['Escenario'] = pd.Categorical(total['Escenario'], categories=orden_escenarios, ordered=True)
total = total.sort_values('Escenario')

# Separar grupos por enfoque
grupos = {
    'NC': total[total['Escenario'].str.endswith('_NC')].copy(),
    'DC': total[total['Escenario'].str.endswith('_DC')].copy(),
    'SC': total[total['Escenario'].str.endswith('_SC')].copy()
}

# === Graficar ===
fig, ax = plt.subplots(figsize=(14, 7))
bars = ax.bar(total['Escenario'], total['SystemInertia'],
              color=[colores_escenarios.get(esc, 'gray') for esc in total['Escenario']],
              edgecolor='black', alpha=0.9)

# Etiquetas numéricas sobre barras
for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02 * total['SystemInertia'].max(),
            f'{h:,.0f}', ha='center', va='bottom', fontsize=9)

# === Curvas de tendencia ===
def plot_line(grupo, color, linestyle):
    y = grupo['SystemInertia'].values
    x = [i for i in range(len(total)) if total['Escenario'].iloc[i] in grupo['Escenario'].values]
    ax.plot(x, y, linestyle, color=color, linewidth=1.5, marker='o')

plot_line(grupos['NC'], 'black', '--')
plot_line(grupos['DC'], 'blue', '-.')
plot_line(grupos['SC'], 'green', ':')

# === Mapeo de barras a escenario ===
bars_dict = {escenario: bar for escenario, bar in zip(total['Escenario'], bars)}

# === Porcentajes de cambio respecto a S0_xx (CORREGIDO) ===
def annotate_percent_change(grupo, base_escenario):
    base_val = grupo.loc[grupo['Escenario'] == base_escenario, 'SystemInertia'].values[0]
    for _, row in grupo.iterrows():
        if row['Escenario'] != base_escenario:
            delta = (row['SystemInertia'] - base_val) / base_val * 100
            bar = bars_dict[row['Escenario']]
            x = bar.get_x() + bar.get_width() / 2
            y = bar.get_height()
            ax.text(x, y + 0.05 * total['SystemInertia'].max(),
                    f'↓{delta:+.1f}%', ha='center', fontsize=9, color='darkred')

# Llamar con los grupos correctos
annotate_percent_change(grupos['NC'], 'EPI_NC')
annotate_percent_change(grupos['DC'], 'EPI_DC')
annotate_percent_change(grupos['SC'], 'EPI_SC')

# === Estética final ===
ax.set_title('Annual Total System Inertia by Scenario', fontsize=16, fontweight='bold')
ax.set_xlabel('Scenario', fontsize=12)
ax.set_ylabel('Total System Inertia (GWs)', fontsize=12)
ax.spines[['top', 'right']].set_visible(False)
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))
plt.xticks(rotation=45)
ax.legend(['NC', 'DC', 'SC'], title='Approach', loc='upper left')
plt.tight_layout()
plt.show()

#%% VALORES MANUALES
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

# === 1. Define values manually ===
data_manual =  {
    'Scenario': ['EPI_NC', 'EPI_DC', 'EPI_SC', 'NZE_NC', 'NZE_DC', 'NZE_SC'],
    'AvailableInertia': [116985.92, 173256.75, 247543.22, 142507.47, 194069.69, 244893.24],
    'RequiredInertia':  [0, 165883, 245280, 0, 174110, 236520]
}

df = pd.DataFrame(data_manual)
order = ['EPI_NC', 'EPI_DC', 'EPI_SC', 'NZE_NC', 'NZE_DC', 'NZE_SC']
df['Scenario'] = pd.Categorical(df['Scenario'], categories=order, ordered=True)
df = df.sort_values('Scenario')

# === Group by approach ===
groups = {
    'NC': df[df['Scenario'].str.endswith('_NC')].copy(),
    'DC': df[df['Scenario'].str.endswith('_DC')].copy(),
    'SC': df[df['Scenario'].str.endswith('_SC')].copy()
}

# === Colors ===
colors_available = {
    'EPI_NC': '#BFD7ED', 'EPI_DC': '#B0D9B1', 'EPI_SC': '#A7C7E7',
    'NZE_NC': '#FFE0B2', 'NZE_DC': '#FFCCBC', 'NZE_SC': '#E2A9BE'
}
color_required = 'grey'

# === Plotting ===
fig, ax = plt.subplots(figsize=(14, 7))

# Base bar: Available inertia
bars_available = ax.bar(
    df['Scenario'],
    df['AvailableInertia'],
    color=[colors_available.get(s, 'gray') for s in df['Scenario']],
    edgecolor='black',
    label='Available Inertia',
    alpha=0.7
)

# Overlay bar: Required inertia
bars_required = ax.bar(
    df['Scenario'],
    df['RequiredInertia'],
    color=color_required,
    edgecolor='black',
    # width=0.5,
    label='Required Inertia',
    alpha=0.7
)

# === Label values ===
for bar in bars_required:
    h = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2, h + 0.01 * df['AvailableInertia'].max(),
        f'{h:,.0f}', ha='center', va='bottom', fontsize=9, color='black'
    )

# === Trend lines ===
def plot_line(grp, column, color, style):
    y = grp[column].values
    x = [i for i in range(len(df)) if df['Scenario'].iloc[i] in grp['Scenario'].values]
    ax.plot(x, y, style, color=color, linewidth=1.5, marker='o')

plot_line(groups['NC'], 'AvailableInertia', 'gray', '--')
plot_line(groups['DC'], 'AvailableInertia', 'blue', '-.')
plot_line(groups['SC'], 'AvailableInertia', 'green', ':')

# === % change vs. EPI baseline ===
def annotate_change(grp, base_scenario):
    base_val = grp.loc[grp['Scenario'] == base_scenario, 'AvailableInertia'].values[0]
    for _, row in grp.iterrows():
        if row['Scenario'] != base_scenario:
            delta = (row['AvailableInertia'] - base_val) / base_val * 100
            bar = bars_available[df['Scenario'] == row['Scenario']][0]
            x = bar.get_x() + bar.get_width() / 2
            y = bar.get_height()
            ax.text(
                x, y + 0.05 * df['AvailableInertia'].max(),
                f'↓{delta:+.1f}%', ha='center', fontsize=9, color='darkred'
            )

annotate_change(groups['NC'], 'EPI_NC')
annotate_change(groups['DC'], 'EPI_DC')
annotate_change(groups['SC'], 'EPI_SC')

# === Final style ===
ax.set_title('Annual Total System Inertia by Scenario', fontsize=16, fontweight='bold')
ax.set_xlabel('Scenario', fontsize=12)
ax.set_ylabel('System Inertia (GWs)', fontsize=12)
ax.spines[['top', 'right']].set_visible(False)
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))
plt.xticks(rotation=45)
ax.legend(title='Inertia Type', loc='upper left')
plt.tight_layout()
plt.show()
#%% MANUAL REQUIRED VS ANUAL
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

# === Datos ===
# estos son los originales en GWs
# data = {
#     'Scenario': ['EPI_NC', 'EPI_DC', 'EPI_SC', 'NZE_NC', 'NZE_DC', 'NZE_SC'],
#     'AvailableInertia': [116985.92, 173256.75, 247543.22, 142507.47, 194069.69, 244893.24],
#     'RequiredInertia':  [0, 165883, 245280, 0, 174110, 236520]
# }
data = {
    'Scenario': ['EPI_NC', 'EPI_DC', 'EPI_SC', 'NZE_NC', 'NZE_DC', 'NZE_SC'],
    'AvailableInertia': [117, 173, 247, 142, 194, 245],
    'RequiredInertia':  [0, 165, 245, 0, 174, 236]
}

df = pd.DataFrame(data)
order = ['EPI_NC', 'EPI_DC', 'EPI_SC', 'NZE_NC', 'NZE_DC', 'NZE_SC']
df['Scenario'] = pd.Categorical(df['Scenario'], categories=order, ordered=True)
df = df.sort_values('Scenario')

# === Calcular diferencia para stacking ===
df['UnusedInertia'] = df['AvailableInertia'] - df['RequiredInertia']

# === Colores personalizados ===
color_required = '#2c3e50'  # azul oscuro elegante
alpha_required = 0.7        # opacidad deseada
color_unused = '#B0D9B1'    # azul claro profesional
alpha_unused = 0.7        # opacidad deseada

# === Crear gráfico ===
fig, ax = plt.subplots(figsize=(12, 7))

bars_required = ax.bar(
    df['Scenario'], df['RequiredInertia'],
    color=color_required, alpha=alpha_required,
    edgecolor='black', label='Required Inertia'
)

bars_unused = ax.bar(
    df['Scenario'], df['UnusedInertia'],
    bottom=df['RequiredInertia'],
    color=color_unused, alpha=alpha_required,
    edgecolor='black', label='Unused Available Inertia'
)

# === Etiquetas numéricas ===
for i in range(len(df)):
    total = df['AvailableInertia'].iloc[i]
    required = df['RequiredInertia'].iloc[i]
    
    # Etiqueta total (en la parte superior)
    ax.text(i, total + 5, f'{int(total):,}', ha='center', fontsize=12, color='black')
    
    # Etiqueta requerida (en la parte inferior de la barra)
    if required > 0:
        ax.text(i, required - 10, f'{int(required):,}', ha='center', fontsize=12, color='white')

# === Títulos y etiquetas ===
ax.set_title('Total System Inertia by Scenario', fontsize=16, fontweight='bold')
ax.set_ylabel('System Inertia (TWs)', fontsize=12)
ax.set_xlabel('Scenario', fontsize=12)
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))
plt.xticks(rotation=45)

# === Eliminar bordes superiores/derechos y agregar leyenda ===
ax.spines[['top', 'right']].set_visible(False)
ax.legend(loc='upper left')
plt.tight_layout()
plt.show()


#%%

# === 2. Cumulative distribution SystemInertia by scenario (Boxplot) ===
# Asegurar orden consistente con gráfico anterior
# colores_escenarios = {
#     'EPI_NC':  '#5fa2d9',  # azul claro
#     'EPI_SC':  '#12507d',  # azul oscuro
#     'EPI_DC':  '#1f77b4',  # azul

#     'NZE_NC':  '#ffb666',  # naranja claro
#     'NZE_SC':  '#b25900',  # naranja oscuro
#     'NZE_DC':  '#ff7f0e',  # naranja

# }

colores_escenarios = {
    'EPI_NC':  '#1f77b4',  # azul claro
    'EPI_SC':  '#1f77b4',  # azul oscuro
    'EPI_DC':  '#1f77b4',  # azul

    'NZE_NC':  '#ff7f0e',  # naranja claro
    'NZE_SC':  '#ff7f0e',  # naranja oscuro
    'NZE_DC':  '#ff7f0e',  # naranja

}

df_concat['Escenario'] = pd.Categorical(df_concat['Escenario'],
                                        categories=orden_escenarios,
                                        ordered=True)

fig, ax = plt.subplots(figsize=(14, 8))
sns.boxplot(data=df_concat, x='Escenario', y='SystemInertia',
            palette=colores_escenarios, fliersize=2, linewidth=1, ax=ax)

ax.set_title('Cumulative Distribution of System Inertia by Scenario', fontsize=26, fontweight='bold')
ax.set_ylabel('System Inertia (GWs)', fontsize=20)
ax.set_xlabel('Scenario', fontsize=20)
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.show()

#%%
import matplotlib.pyplot as plt
import seaborn as sns

# Estilo general limpio
sns.set_theme(style="whitegrid", font_scale=1.2)

# Categoría ordenada
df_concat['Escenario'] = pd.Categorical(df_concat['Escenario'],
                                        categories=orden_escenarios,
                                        ordered=True)

fig, ax = plt.subplots(figsize=(14, 8))

# Boxplot sin outliers visuales
sns.boxplot(data=df_concat, x='Escenario', y='SystemInertia',
            palette=colores_escenarios, fliersize=0, linewidth=1.2, ax=ax)

# Stripplot con transparencia para evitar saturación
sns.stripplot(data=df_concat, x='Escenario', y='SystemInertia',
              color='black', size=3.5, alpha=0.25, jitter=0.3, ax=ax)

# Título claro y directo
ax.set_title('Distribución acumulativa de la Inercia del Sistema por Escenario',
             fontsize=22, fontweight='bold', pad=20)

# Ejes bien etiquetados
ax.set_xlabel('Escenario', fontsize=18, labelpad=10)
ax.set_ylabel('Inercia del Sistema (GWs)', fontsize=18, labelpad=10)

# Ejes más limpios y legibles
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='x', visible=False)  # solo grid horizontal si es necesario

plt.tight_layout()
plt.show()


#%%

import matplotlib.pyplot as plt
import seaborn as sns

# Estilo general elegante
sns.set_theme(style="whitegrid", font_scale=1.2)

# Asegurar el orden
df_concat['Escenario'] = pd.Categorical(df_concat['Escenario'],
                                        categories=orden_escenarios,
                                        ordered=True)

fig, ax = plt.subplots(figsize=(14, 8))

# Boxplot principal
sns.boxplot(data=df_concat, x='Escenario', y='SystemInertia',
            palette=colores_escenarios, fliersize=0, linewidth=1.2, ax=ax)

# Agregar los puntos coloreados uno por uno
for escenario in orden_escenarios:
    color = colores_escenarios.get(escenario, 'gray')
    data_esc = df_concat[df_concat['Escenario'] == escenario]
    x_pos = orden_escenarios.index(escenario)
    
    sns.stripplot(x=[x_pos]*len(data_esc),
                  y=data_esc['SystemInertia'],
                  color=color,
                  size=3.5,
                  alpha=0.25,  # transparencia
                  jitter=0.25,
                  ax=ax)

# Título y etiquetas limpias
ax.set_title('Distribución acumulativa de la Inercia del Sistema por Escenario',
             fontsize=22, fontweight='bold', pad=20)
ax.set_xlabel('Escenario', fontsize=18, labelpad=10)
ax.set_ylabel('Inercia del Sistema (GWs)', fontsize=18, labelpad=10)

# Detalles de estilo
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='x', visible=False)

plt.tight_layout()
plt.show()


#%%
# === 3. timeseries Daily Average System Inertia with Range (Min–Max) by Scenario Group ===
import matplotlib.pyplot as plt
import pandas as pd

# Asegurarse que 'index' sea datetime
df_concat['index'] = pd.to_datetime(df_concat['index'])

# Extraer solo la fecha para el promedio diario
df_concat['Fecha'] = df_concat['index'].dt.date

# Agrupar por escenario y fecha para obtener promedio, mínimo y máximo diario
df_stats = df_concat.groupby(['Escenario', 'Fecha'])['SystemInertia'].agg(['mean', 'min', 'max']).reset_index()

# Grupos de escenarios
grupos_escenarios = {
    'Non Constrained Approach': ['EPI_NC', 'NZE_NC'],
    'Static Constrained Approach':   ['EPI_SC', 'NZE_SC'],
    'Dynamic Constrained Approach':   ['EPI_DC', 'NZE_DC'],
}

# Crear figura y ejes
fig, axs = plt.subplots(3, 1, figsize=(30, 15), sharex=True)

# Plot settings
for ax, (group_title, esc_list) in zip(axs, grupos_escenarios.items()):
    for esc in esc_list:
        df_esc = df_stats[df_stats['Escenario'] == esc]
        if df_esc.empty:
            continue
        color = colores_escenarios.get(esc, 'gray')

        ax.fill_between(df_esc['Fecha'], df_esc['min'], df_esc['max'], color=color, alpha=0.2)
        ax.plot(df_esc['Fecha'], df_esc['mean'], label=esc, color=color, linewidth=1.8)

    ax.set_title(group_title, fontsize=14, fontweight='bold')
    ax.set_ylabel('Daily System Inertia (MW·s)', fontsize=12)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=10)
    ax.legend(loc='upper right', fontsize=9, ncol=3, frameon=False)

# Etiqueta común en el eje X
axs[-1].set_xlabel('Date', fontsize=12)

# Título principal
fig.suptitle('Daily Average System Inertia with Range (Min–Max) by Approach',
             fontsize=16, fontweight='bold', y=0.995)

plt.tight_layout()
plt.subplots_adjust(top=0.94)
plt.show()


#%%
# === 3.1 (eje y fijo para todos de 0 a 30) timeseries Daily Average System Inertia with Range (Min–Max) by Scenario Group ===

import matplotlib.pyplot as plt
import pandas as pd

# Asegurarse que 'index' sea datetime
df_concat['index'] = pd.to_datetime(df_concat['index'])

# Extraer solo la fecha para el promedio diario
df_concat['Fecha'] = df_concat['index'].dt.date

# Agrupar por escenario y fecha para obtener promedio, mínimo y máximo diario
df_stats = df_concat.groupby(['Escenario', 'Fecha'])['SystemInertia'].agg(['mean', 'min', 'max']).reset_index()

# Grupos de escenarios
grupos_escenarios = {
    'Non Constrained Approach': ['EPI_NC', 'NZE_NC'],
    'Static Constrained Approach':   ['EPI_SC', 'NZE_SC'],
    'Dynamic Constrained Approach':   ['EPI_DC', 'NZE_DC'],
}

# Crear figura y ejes
fig, axs = plt.subplots(3, 1, figsize=(30, 15), sharex=True)

# Plot settings
for ax, (group_title, esc_list) in zip(axs, grupos_escenarios.items()):
    for esc in esc_list:
        df_esc = df_stats[df_stats['Escenario'] == esc]
        if df_esc.empty:
            continue
        color = colores_escenarios.get(esc, 'gray')

        ax.fill_between(df_esc['Fecha'], df_esc['min'], df_esc['max'], color=color, alpha=0.2)
        ax.plot(df_esc['Fecha'], df_esc['mean'], label=esc, color=color, linewidth=1.8)

    ax.set_title(group_title, fontsize=14, fontweight='bold')
    ax.set_ylabel('Daily System Inertia (MW·s)', fontsize=12)
    ax.set_ylim(0, 30)  # Fijar eje Y de 0 a 30
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=10)
    ax.legend(loc='upper right', fontsize=9, ncol=3, frameon=False)

# Etiqueta común en el eje X
axs[-1].set_xlabel('Date', fontsize=12)

# Título principal
fig.suptitle('Daily Average System Inertia with Range (Min–Max) by Approach',
             fontsize=16, fontweight='bold', y=0.995)

plt.tight_layout()
plt.subplots_adjust(top=0.94)
plt.show()
#%%
# === 3.2 (eje y fijo para todos y linea mas delgada) timeseries Daily Average System Inertia with Range (Min–Max) by Scenario Group ===

import matplotlib.pyplot as plt
import pandas as pd

# Asegurarse que 'index' sea datetime
df_concat['index'] = pd.to_datetime(df_concat['index'])

# Extraer solo la fecha para el promedio diario
df_concat['Fecha'] = df_concat['index'].dt.date

# Agrupar por escenario y fecha para obtener promedio, mínimo y máximo diario
df_stats = df_concat.groupby(['Escenario', 'Fecha'])['SystemInertia'].agg(['mean', 'min', 'max']).reset_index()

# Grupos de escenarios
grupos_escenarios = {
    'Non Constrained Approach': ['EPI_NC', 'NZE_NC'],
    'Dynamic Constrained Approach':   ['EPI_DC', 'NZE_DC'],
    'Static Constrained Approach':   ['EPI_SC', 'NZE_SC'],
}

# Crear figura y ejes
fig, axs = plt.subplots(3, 1, figsize=(30, 15), sharex=True)

# Estética del gráfico
for ax, (group_title, esc_list) in zip(axs, grupos_escenarios.items()):
    for esc in esc_list:
        df_esc = df_stats[df_stats['Escenario'] == esc]
        if df_esc.empty:
            continue
        color = colores_escenarios.get(esc, 'gray')

        # Banda min–max
        ax.fill_between(df_esc['Fecha'], df_esc['min'], df_esc['max'], color=color, alpha=0.2)

        # Línea de promedio
        ax.plot(df_esc['Fecha'], df_esc['mean'], label=esc, color=color, linewidth=1)

    ax.set_title(group_title, fontsize=26, fontweight='bold')
    ax.set_ylabel('Daily System Inertia (GW·s)', fontsize=26)
    ax.set_ylim(0, 35)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=22)
    ax.legend(loc='upper left', fontsize=22, ncol=3, frameon=False)

# Etiqueta común en el eje X
axs[-1].set_xlabel('Date', fontsize=26)

# Título general
fig.suptitle('Daily Average System Inertia with Range (Min–Max) by Approach',
             fontsize=26, fontweight='bold', y=0.995)

plt.tight_layout()
plt.subplots_adjust(top=0.94)
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
#%%
# === 4.1 mejor todo agrupado Mapa de calor: SystemInertia medio por zona y hora ===
# Asegurar columnas auxiliares
df_concat['Hora'] = pd.to_datetime(df_concat['index']).dt.hour
df_concat['Approach'] = df_concat['Escenario'].apply(lambda x: x.split('_')[-1] if '_' in x else 'NC')

# Agrupar para el heatmap: promedio por hora y escenario
pivot_table = df_concat.groupby(['Approach', 'Escenario', 'Hora'])['SystemInertia'].mean().reset_index()

# Plot: un heatmap por approach
import seaborn as sns
import matplotlib.pyplot as plt

approaches = ['NC', 'DC', 'SC']
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(20, 6), sharey=True)

for i, approach in enumerate(approaches):
    df_ap = pivot_table[pivot_table['Approach'] == approach]
    heatmap_data = df_ap.pivot(index='Escenario', columns='Hora', values='SystemInertia')

    sns.heatmap(heatmap_data, ax=axs[i], cmap='viridis', annot=False,
                vmin=pivot_table['SystemInertia'].mean() - 2*pivot_table['SystemInertia'].std(),
                vmax=pivot_table['SystemInertia'].mean() + 2*pivot_table['SystemInertia'].std(),
                cbar=i==2, cbar_kws={'label': 'SystemInertia (MWs)'})
    
    axs[i].set_title(f'{approach} Approach', fontsize=16)
    axs[i].set_xlabel('Hour of Day')
    if i == 0:
        axs[i].set_ylabel('Scenario')

plt.suptitle('Average System Inertia by Hour and Scenario – by Approach', fontsize=18, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
#%%MAPA DE CALOR PARA PRESENTACION
# Asegurar columnas auxiliares
df_concat['Hora'] = pd.to_datetime(df_concat['index']).dt.hour
df_concat['Approach'] = df_concat['Escenario'].apply(lambda x: x.split('_')[-1] if '_' in x else 'NC')
df_concat['ScenarioBase'] = df_concat['Escenario'].apply(lambda x: x.split('_')[0])  # EPI o NZE

# Agrupar para el heatmap: promedio por hora y escenario base
pivot_table = df_concat.groupby(['Approach', 'ScenarioBase', 'Hora'])['SystemInertia'].mean().reset_index()

# Plot: un heatmap por approach
import seaborn as sns
import matplotlib.pyplot as plt

approaches = ['NC', 'DC', 'SC']
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(20, 6), sharey=True)

for i, approach in enumerate(approaches):
    df_ap = pivot_table[pivot_table['Approach'] == approach]
    heatmap_data = df_ap.pivot(index='ScenarioBase', columns='Hora', values='SystemInertia')

    sns.heatmap(
        heatmap_data, ax=axs[i], cmap='viridis', annot=False,
        vmin=pivot_table['SystemInertia'].mean() - 2*pivot_table['SystemInertia'].std(),
        vmax=pivot_table['SystemInertia'].mean() + 2*pivot_table['SystemInertia'].std(),
        cbar=i == 2, cbar_kws={'label': 'SystemInertia (MWs)'}
    )
    
    axs[i].set_title(f'{approach} Approach', fontsize=16)
    axs[i].set_xlabel('Hour of Day')
    if i == 0:
        axs[i].set_ylabel('Scenario')

plt.suptitle('Average System Inertia by Hour and Scenario – by Approach', fontsize=18, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
#%%
import seaborn as sns
import matplotlib.pyplot as plt

# Columnas auxiliares
df_concat['Hora'] = pd.to_datetime(df_concat['index']).dt.hour
df_concat['Approach'] = df_concat['Escenario'].apply(lambda x: x.split('_')[-1] if '_' in x else 'NC')
df_concat['ScenarioBase'] = df_concat['Escenario'].apply(lambda x: x.split('_')[0])  # EPI o NZE

# Agrupación
pivot_table = df_concat.groupby(['Approach', 'ScenarioBase', 'Hora'])['SystemInertia'].mean().reset_index()

# Plot
approaches = ['NC', 'DC', 'SC']
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(22, 6), sharey=True)

# Escenarios base ordenados
scenario_order = ['EPI', 'NZE']

for i, approach in enumerate(approaches):
    df_ap = pivot_table[pivot_table['Approach'] == approach]
    heatmap_data = df_ap.pivot(index='ScenarioBase', columns='Hora', values='SystemInertia').reindex(scenario_order)

    sns.heatmap(
        heatmap_data,
        ax=axs[i],
        cmap='crest',  # paleta profesional
        annot=False,
        linewidths=0.3,
        linecolor='gray',
        cbar=i == 2,
        cbar_kws={'label': 'System Inertia (MWs)'},
        vmin=pivot_table['SystemInertia'].mean() - 2 * pivot_table['SystemInertia'].std(),
        vmax=pivot_table['SystemInertia'].mean() + 2 * pivot_table['SystemInertia'].std()
    )

    axs[i].set_title(f'{approach} Approach', fontsize=16, fontweight='bold')
    axs[i].set_xlabel('Hour of Day', fontsize=13)
    axs[i].set_xticks(range(0, 24, 3))
    axs[i].set_xticklabels(range(0, 24, 3), fontsize=11)
    axs[i].set_yticklabels(scenario_order, rotation=0, fontsize=12)
    axs[i].tick_params(axis='y', labelsize=12)
    axs[i].tick_params(axis='x', labelsize=11)
    axs[i].set_ylabel('Scenario', fontsize=13)

# Ajustes globales
plt.suptitle('Average System Inertia by Hour and Scenario – by Approach', fontsize=20, fontweight='bold')
plt.subplots_adjust(left=0.05, right=0.95, top=0.88, bottom=0.1, wspace=0.15)
plt.tight_layout()
plt.show()



#%%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Asegurar columnas necesarias
df_concat['Hour'] = pd.to_datetime(df_concat['index']).dt.hour
df_concat['Approach'] = df_concat['Escenario'].apply(lambda x: x.split('_')[-1] if '_' in x else 'NC')
df_concat['NCScenario'] = df_concat['Escenario'].apply(lambda x: x.split('_')[0])

# Agrupar por Approach, Escenario y Hora
df_plot = df_concat.groupby(['Approach', 'NCScenario', 'Hour'])['SystemInertia'].mean().reset_index()

# Escenarios base en orden
escenarios_base = sorted(df_plot['NCScenario'].unique(), key=lambda x: int(x[1:]) if x[1:].isdigit() else x)
df_plot['NCScenario'] = pd.Categorical(df_plot['NCScenario'], categories=escenarios_base[::-1], ordered=True)

# === NUEVO: Mapear escenarios a posiciones numéricas para controlar el espaciado ===
escenarios_ordenados = df_plot['NCScenario'].cat.categories.tolist()
escenario_a_pos = {esc: i * 0.8 for i, esc in enumerate(escenarios_ordenados)}  # puedes ajustar el 0.8
df_plot['ScenarioPos'] = df_plot['NCScenario'].map(escenario_a_pos)

# Rango para colores y tamaños
vmin = df_plot['SystemInertia'].mean() - 2 * df_plot['SystemInertia'].std()
vmax = df_plot['SystemInertia'].mean() + 2 * df_plot['SystemInertia'].std()
cmap = plt.get_cmap("viridis")

# Subplots: uno por approach
approaches = ['NC', 'DC', 'SC']
fig, axs = plt.subplots(1, 3, figsize=(15, 3), sharey=True)

for i, app in enumerate(approaches):
    ax = axs[i]
    df_app = df_plot[df_plot['Approach'] == app]

    for _, row in df_app.iterrows():
        color = cmap((row['SystemInertia'] - vmin) / (vmax - vmin))
        size = 200 + 1000 * (row['SystemInertia'] - vmin) / (vmax - vmin)
        ax.scatter(row['Hour'], row['ScenarioPos'], color=color, s=size, edgecolors='k', alpha=0.9)

    ax.set_title(f'{app} Approach', fontsize=16)
    ax.set_xlabel("Hour of the day", fontsize=14)
    ax.set_xlim(-1, 24)
    ax.set_xticks(np.arange(0, 25, 3))
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_yticks(list(escenario_a_pos.values()))
    ax.set_yticklabels(escenario_a_pos.keys(), fontsize=12)

axs[0].set_ylabel("Scenarios", fontsize=14)

# === Colorbar fuera del área de los plots ===
cbar_ax = fig.add_axes([0.97, 0.25, 0.015, 0.5])  # [left, bottom, width, height]
norm = plt.Normalize(vmin=vmin, vmax=vmax)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label('SystemInertia (GWs)', fontsize=14)
cbar.ax.tick_params(labelsize=12)

# Título global
fig.suptitle("System Inertia Heat Map by Scenario and Approach", fontsize=20, fontweight='bold')
plt.tight_layout(rect=[0, 0, 0.97, 0.93])
plt.show()

