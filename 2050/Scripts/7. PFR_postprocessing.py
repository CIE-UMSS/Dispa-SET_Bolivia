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
    # df = results['OutputPFR_Available'].copy().to_frame(name='OutputPFR_Available')
    
    # Sumar todas las columnas de PFR por fila para obtener una sola serie
    PFR = results['OutputPrimaryReserve_Available'].sum(axis=1).to_frame(name='PFR_Available')
    
    # Verificar si está vacío
    if PFR.empty:
        # Crear un DataFrame vacío con ceros y el mismo índice de tiempo que otra variable (por ejemplo, OutputPower)
        time_index = results['OutputPower'].index if 'OutputPower' in results else pd.date_range("2022-01-01", periods=8760, freq='H')
        df = pd.DataFrame({'PFR_Available': np.zeros(len(time_index))}, index=time_index)
    else:
        # Si no está vacío, asegurar que sea DataFrame con el nombre correcto
        df = PFR.copy()
        if isinstance(df, pd.Series):
            df = df.to_frame(name='PFR_Available')
        elif isinstance(df, pd.DataFrame):
            df.columns = ['PFR_Available']  # Asegura el nombre correcto de la columna

    # Asegurar que el índice es datetime
    df.index = pd.to_datetime(df.index)

    # Convertir el índice a serie para trabajar con .dt
    index_series = df.index.to_series()

    df['Fecha'] = index_series.dt.date
    df['Hora'] = index_series.dt.hour
    df['Periodo'] = index_series.map(get_season)
    df['Escenario'] = nombre

    # Renombrar la columna '0' a 'PFR_Available'
    df = df.rename(columns={df.columns[0]: 'PFR_Available'})
    
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

# Ahora df_concat ya tiene columnas: ['index', 'PFR_Available', 'Fecha', 'Hora', 'Periodo', 'Escenario']

#%%
# === 1. Annual total PFR_Available by scenario ===

# Generar orden deseado por escenario (S0 a S6) y enfoque (NC, DC, SC)
orden_escenarios = ['EPI_NC', 'EPI_DC', 'EPI_SC', 'NZE_NC', 'NZE_DC', 'NZE_SC']

# Agrupar y ordenar
total = df_concat.groupby('Escenario')['PFR_Available'].sum().reset_index()
total['Escenario'] = pd.Categorical(total['Escenario'], categories=orden_escenarios, ordered=True)
total = total.sort_values('Escenario')

# Graficar
fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(total['Escenario'], total['PFR_Available'],
              color=[colores_escenarios[esc] for esc in total['Escenario']],
              edgecolor='black', alpha=0.9)

# Añadir valores sobre las barras
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.02 * total['PFR_Available'].max(),
            f'{height:,.0f}', ha='center', va='bottom', fontsize=9)

# Estética del gráfico
ax.set_xlabel('Scenario', fontsize=12)
ax.set_ylabel('Total PFR (MW)', fontsize=12)
ax.set_title('Annual Total PFR by Scenario', fontsize=14, fontweight='bold')
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.show()

#%%
# === 1.1 Annual total PFR_Available by scenario con curvas tendencia ===
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

# === Preparar datos ===
orden_escenarios = ['EPI_NC', 'EPI_DC', 'EPI_SC', 'NZE_NC', 'NZE_DC', 'NZE_SC']
total = df_concat.groupby('Escenario')['PFR_Available'].sum().reset_index()
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
bars = ax.bar(total['Escenario'], total['PFR_Available'],
              color=[colores_escenarios.get(esc, 'gray') for esc in total['Escenario']],
              edgecolor='black', alpha=0.9)

# Etiquetas numéricas sobre barras
for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02 * total['PFR_Available'].max(),
            f'{h:,.0f}', ha='center', va='bottom', fontsize=9)

# === Curvas de tendencia ===
def plot_line(grupo, color, linestyle):
    y = grupo['PFR_Available'].values
    x = [i for i in range(len(total)) if total['Escenario'].iloc[i] in grupo['Escenario'].values]
    ax.plot(x, y, linestyle, color=color, linewidth=1.5, marker='o')

plot_line(grupos['NC'], 'black', '--')
plot_line(grupos['DC'], 'blue', '-.')
plot_line(grupos['SC'], 'green', ':')

# === Mapeo de barras a escenario ===
bars_dict = {escenario: bar for escenario, bar in zip(total['Escenario'], bars)}

# === Porcentajes de cambio respecto a S0_xx (CORREGIDO) ===
def annotate_percent_change(grupo, base_escenario):
    base_val = grupo.loc[grupo['Escenario'] == base_escenario, 'PFR_Available'].values[0]
    for _, row in grupo.iterrows():
        if row['Escenario'] != base_escenario:
            delta = (row['PFR_Available'] - base_val) / base_val * 100
            bar = bars_dict[row['Escenario']]
            x = bar.get_x() + bar.get_width() / 2
            y = bar.get_height()
            ax.text(x, y + 0.05 * total['PFR_Available'].max(),
                    f'↓{delta:+.1f}%', ha='center', fontsize=9, color='darkred')

# Llamar con los grupos correctos
annotate_percent_change(grupos['NC'], 'EPI_NC')
annotate_percent_change(grupos['DC'], 'EPI_DC')
annotate_percent_change(grupos['SC'], 'EPI_SC')

# === Estética final ===
ax.set_title('Annual Total PFR by Scenario', fontsize=16, fontweight='bold')
ax.set_xlabel('Scenario', fontsize=12)
ax.set_ylabel('Total PFR (MW)', fontsize=12)
ax.spines[['top', 'right']].set_visible(False)
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))
plt.xticks(rotation=45)
ax.legend(['NC', 'DC', 'SC'], title='Approach', loc='upper left')
plt.tight_layout()
plt.show()
#%% MANUAL REQUIRED VS ANUAL

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

# === Datos ===
# datos originales en MW
# data = {
#     'Scenario': ['EPI_NC', 'EPI_DC', 'EPI_SC', 'NZE_NC', 'NZE_DC', 'NZE_SC'],
#     'AvailablePFR': [0, 764847.56, 870300.25, 0, 894648.03, 1023672.25],
#     'RequiredPFR':  [0, 751520.09, 870181.83, 0, 727762.11, 923708.75]
# }
data = {
    'Scenario': ['EPI_NC', 'EPI_DC', 'EPI_SC', 'NZE_NC', 'NZE_DC', 'NZE_SC'],
    'AvailablePFR': [0, 764.85, 870.30, 0, 894.65, 1023.67],
    'RequiredPFR':  [0, 751.52, 870.18, 0, 727.76, 923.71]
}

df = pd.DataFrame(data)
order = ['EPI_NC', 'EPI_DC', 'EPI_SC', 'NZE_NC', 'NZE_DC', 'NZE_SC']
df['Scenario'] = pd.Categorical(df['Scenario'], categories=order, ordered=True)
df = df.sort_values('Scenario')

# === Calcular diferencia para stacking ===
df['UnusedPFR'] = df['AvailablePFR'] - df['RequiredPFR']

# === Colores personalizados ===
color_required = '#2c3e50'  # azul oscuro elegante
alpha_required = 0.7        # opacidad deseada
color_unused = '#20B2AA'   # azul claro profesional

# === Crear gráfico ===
fig, ax = plt.subplots(figsize=(12, 7))

bars_required = ax.bar(
    df['Scenario'], df['RequiredPFR'],
    color=color_required, alpha=alpha_required,
    edgecolor='black', label='Required PFR'
)

bars_unused = ax.bar(
    df['Scenario'], df['UnusedPFR'],
    bottom=df['RequiredPFR'],
    color=color_unused, alpha=alpha_required,
    edgecolor='black', label='Unused Available PFR'
)

# === Etiquetas numéricas ===
for i in range(len(df)):
    total = df['AvailablePFR'].iloc[i]
    required = df['RequiredPFR'].iloc[i]
    
    # Etiqueta total (en la parte superior)
    ax.text(i, total + 10, f'{total:.2f}', ha='center', fontsize=12, color='black')
    
    # Etiqueta requerida (en la parte inferior de la barra)
    if required > 0:
        ax.text(i, required - 30, f'{required:.2f}', ha='center', fontsize=12, color='white')

# === Títulos y etiquetas ===
ax.set_title('Total PFR by Scenario', fontsize=16, fontweight='bold')
ax.set_ylabel('PFR (GWh)', fontsize=12)
ax.set_xlabel('Scenario', fontsize=12)
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))
plt.xticks(rotation=45)

# === Eliminar bordes superiores/derechos y agregar leyenda ===
ax.spines[['top', 'right']].set_visible(False)
ax.legend(loc='upper left')
plt.tight_layout()
plt.show()



#%%



# === 2. Cumulative distribution PFR_Available by scenario (Boxplot) ===
# Asegurar orden consistente con gráfico anterior
colores_escenarios = {
    'EPI_NC':  '#5fa2d9',  # azul claro
    'EPI_SC':  '#12507d',  # azul oscuro
    'EPI_DC':  '#1f77b4',  # azul

    'NZE_NC':  '#ffb666',  # naranja claro
    'NZE_SC':  '#b25900',  # naranja oscuro
    'NZE_DC':  '#ff7f0e',  # naranja

}
df_concat['Escenario'] = pd.Categorical(df_concat['Escenario'],
                                        categories=orden_escenarios,
                                        ordered=True)

fig, ax = plt.subplots(figsize=(14, 8))
sns.boxplot(data=df_concat, x='Escenario', y='PFR_Available',
            palette=colores_escenarios, fliersize=2, linewidth=1, ax=ax)

ax.set_title('Cumulative Distribution of System PFR by Scenario', fontsize=26, fontweight='bold')
ax.set_ylabel('System PFR (GWs)', fontsize=20)
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
sns.boxplot(data=df_concat, x='Escenario', y='PFR_Available',
            palette=colores_escenarios, fliersize=0, linewidth=1.2, ax=ax)

# Stripplot con transparencia para evitar saturación
sns.stripplot(data=df_concat, x='Escenario', y='PFR_Available',
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
sns.boxplot(data=df_concat, x='Escenario', y='PFR_Available',
            palette=colores_escenarios, fliersize=0, linewidth=1.2, ax=ax)

# Agregar los puntos coloreados uno por uno
for escenario in orden_escenarios:
    color = colores_escenarios.get(escenario, 'gray')
    data_esc = df_concat[df_concat['Escenario'] == escenario]
    x_pos = orden_escenarios.index(escenario)
    
    sns.stripplot(x=[x_pos]*len(data_esc),
                  y=data_esc['PFR_Available'],
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
# === 3. timeseries Daily Average System PFR with Range (Min–Max) by Scenario Group ===
import matplotlib.pyplot as plt
import pandas as pd

# Asegurarse que 'index' sea datetime
df_concat['index'] = pd.to_datetime(df_concat['index'])

# Extraer solo la fecha para el promedio diario
df_concat['Fecha'] = df_concat['index'].dt.date

# Agrupar por escenario y fecha para obtener promedio, mínimo y máximo diario
df_stats = df_concat.groupby(['Escenario', 'Fecha'])['PFR_Available'].agg(['mean', 'min', 'max']).reset_index()

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
    ax.set_ylabel('Daily System PFR (MW·s)', fontsize=12)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=10)
    ax.legend(loc='upper right', fontsize=9, ncol=3, frameon=False)

# Etiqueta común en el eje X
axs[-1].set_xlabel('Date', fontsize=12)

# Título principal
fig.suptitle('Daily Average System PFR with Range (Min–Max) by Approach',
             fontsize=16, fontweight='bold', y=0.995)

plt.tight_layout()
plt.subplots_adjust(top=0.94)
plt.show()


#%%
# === 3.1 (eje y fijo para todos de 0 a 30) timeseries Daily Average System PFR with Range (Min–Max) by Scenario Group ===

import matplotlib.pyplot as plt
import pandas as pd

# Asegurarse que 'index' sea datetime
df_concat['index'] = pd.to_datetime(df_concat['index'])

# Extraer solo la fecha para el promedio diario
df_concat['Fecha'] = df_concat['index'].dt.date

# Agrupar por escenario y fecha para obtener promedio, mínimo y máximo diario
df_stats = df_concat.groupby(['Escenario', 'Fecha'])['PFR_Available'].agg(['mean', 'min', 'max']).reset_index()

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
    ax.set_ylabel('Daily System PFR (MW·s)', fontsize=12)
    ax.set_ylim(0, 30)  # Fijar eje Y de 0 a 30
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=10)
    ax.legend(loc='upper right', fontsize=9, ncol=3, frameon=False)

# Etiqueta común en el eje X
axs[-1].set_xlabel('Date', fontsize=12)

# Título principal
fig.suptitle('Daily Average System PFR with Range (Min–Max) by Approach',
             fontsize=16, fontweight='bold', y=0.995)

plt.tight_layout()
plt.subplots_adjust(top=0.94)
plt.show()
#%%
# === 3.2 (eje y fijo para todos y linea mas delgada) timeseries Daily Average System PFR with Range (Min–Max) by Scenario Group ===

import matplotlib.pyplot as plt
import pandas as pd

# Asegurarse que 'index' sea datetime
df_concat['index'] = pd.to_datetime(df_concat['index'])

# Extraer solo la fecha para el promedio diario
df_concat['Fecha'] = df_concat['index'].dt.date

# Agrupar por escenario y fecha para obtener promedio, mínimo y máximo diario
df_stats = df_concat.groupby(['Escenario', 'Fecha'])['PFR_Available'].agg(['mean', 'min', 'max']).reset_index()

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
    ax.set_ylabel('Daily System PFR (GW·s)', fontsize=26)
    ax.set_ylim(0, 35)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=22)
    ax.legend(loc='upper left', fontsize=22, ncol=3, frameon=False)

# Etiqueta común en el eje X
axs[-1].set_xlabel('Date', fontsize=26)

# Título general
fig.suptitle('Daily Average System PFR with Range (Min–Max) by Approach',
             fontsize=26, fontweight='bold', y=0.995)

plt.tight_layout()
plt.subplots_adjust(top=0.94)
plt.show()


#%%
# === 4. Mapa de calor: PFR_Available medio por zona y hora ===
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
vmin = df_concat['PFR_Available'].mean() - 2 * df_concat['PFR_Available'].std()
vmax = df_concat['PFR_Available'].mean() + 2 * df_concat['PFR_Available'].std()

# Dibujar heatmaps
for i, esc in enumerate(escenarios):
    ax = axs[i]
    df_esc = df_concat[df_concat['Escenario'] == esc]
    df_heatmap = df_esc.groupby('Hora')['PFR_Available'].mean().to_frame().T  # Solo 1 fila

    sns.heatmap(df_heatmap, cmap='viridis', linewidths=0.4, annot=True, fmt=".0f",
                ax=ax, cbar=i == 0, cbar_ax=cbar_ax if i == 0 else None,
                vmin=vmin, vmax=vmax, annot_kws={'fontsize': 14})

    ax.set_title(f'Escenario: {esc}', fontsize=16, fontweight='bold')
    ax.set_xlabel('Hora del día', fontsize=14)
    ax.set_ylabel('PFR_Available', fontsize=14)
    ax.tick_params(labelsize=12)
    ax.set_yticklabels([''])  # Quitar etiquetas del eje Y ya que es solo una fila

# Ajustar colorbar
cbar_ax.tick_params(labelsize=12)
cbar_ax.set_ylabel('PFR_Available (MWh)', fontsize=16, labelpad=20)

# Título global
fig.suptitle('PFR_Available medio por hora del día – por escenario',
             fontsize=20, fontweight='bold', y=1.02)

plt.tight_layout()
plt.subplots_adjust(right=0.92, top=0.93)
plt.show()
#%%
# === 4.1 mejor todo agrupado Mapa de calor: PFR_Available medio por zona y hora ===
# Asegurar columnas auxiliares
df_concat['Hora'] = pd.to_datetime(df_concat['index']).dt.hour
df_concat['Approach'] = df_concat['Escenario'].apply(lambda x: x.split('_')[-1] if '_' in x else 'NC')

# Agrupar para el heatmap: promedio por hora y escenario
pivot_table = df_concat.groupby(['Approach', 'Escenario', 'Hora'])['PFR_Available'].mean().reset_index()

# Plot: un heatmap por approach
import seaborn as sns
import matplotlib.pyplot as plt

approaches = ['NC', 'DC', 'SC']
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(20, 6), sharey=True)

for i, approach in enumerate(approaches):
    df_ap = pivot_table[pivot_table['Approach'] == approach]
    heatmap_data = df_ap.pivot(index='Escenario', columns='Hora', values='PFR_Available')

    sns.heatmap(heatmap_data, ax=axs[i], cmap='viridis', annot=False,
                vmin=pivot_table['PFR_Available'].mean() - 2*pivot_table['PFR_Available'].std(),
                vmax=pivot_table['PFR_Available'].mean() + 2*pivot_table['PFR_Available'].std(),
                cbar=i==2, cbar_kws={'label': 'PFR_Available (MWs)'})
    
    axs[i].set_title(f'{approach} Approach', fontsize=16)
    axs[i].set_xlabel('Hour of Day')
    if i == 0:
        axs[i].set_ylabel('Scenario')

plt.suptitle('Average System PFR by Hour and Scenario – by Approach', fontsize=18, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
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
df_plot = df_concat.groupby(['Approach', 'NCScenario', 'Hour'])['PFR_Available'].mean().reset_index()

# Escenarios base en orden
escenarios_base = sorted(df_plot['NCScenario'].unique(), key=lambda x: int(x[1:]) if x[1:].isdigit() else x)
df_plot['NCScenario'] = pd.Categorical(df_plot['NCScenario'], categories=escenarios_base[::-1], ordered=True)

# === NUEVO: Mapear escenarios a posiciones numéricas para controlar el espaciado ===
escenarios_ordenados = df_plot['NCScenario'].cat.categories.tolist()
escenario_a_pos = {esc: i * 0.8 for i, esc in enumerate(escenarios_ordenados)}  # puedes ajustar el 0.8
df_plot['ScenarioPos'] = df_plot['NCScenario'].map(escenario_a_pos)

# Rango para colores y tamaños
vmin = df_plot['PFR_Available'].mean() - 2 * df_plot['PFR_Available'].std()
vmax = df_plot['PFR_Available'].mean() + 2 * df_plot['PFR_Available'].std()
cmap = plt.get_cmap("viridis")

# Subplots: uno por approach
approaches = ['NC', 'DC', 'SC']
fig, axs = plt.subplots(1, 3, figsize=(15, 3), sharey=True)

for i, app in enumerate(approaches):
    ax = axs[i]
    df_app = df_plot[df_plot['Approach'] == app]

    for _, row in df_app.iterrows():
        color = cmap((row['PFR_Available'] - vmin) / (vmax - vmin))
        size = 200 + 1000 * (row['PFR_Available'] - vmin) / (vmax - vmin)
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
cbar.set_label('PFR_Available (GWs)', fontsize=14)
cbar.ax.tick_params(labelsize=12)

# Título global
fig.suptitle("System PFR Heat Map by Scenario and Approach", fontsize=20, fontweight='bold')
plt.tight_layout(rect=[0, 0, 0.97, 0.93])
plt.show()

