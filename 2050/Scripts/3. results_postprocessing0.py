# -*- coding: utf-8 -*-
"""
Created on Tue May 20 15:33:53 2025

@author: navia
"""

import matplotlib.pyplot as plt
import numpy as np

# Datos de referencia
parameters = {
    "System inertia (GWs)": {
        "EPI": [71.5, 71.5, 71.5],
        "NZE": [53.0, 50.4, 42.7]
    },
    "FFR (GW)": {
        "EPI": [0.00, 0.00, 0.00],
        "NZE": [0.00, 0.00, 1.58]
    },
    "PFR (GW)": {
        "EPI": [1.5, 1.5, 1.5],
        "NZE": [1.5, 1.5, 1.5]
    },
    "Spillage (TWh)": {
        "EPI": [0.1, 0.1, 0.1],
        "NZE": [1.3, 1.3, 4.4]
    },
    "Emissions (ktCO2)": {
        "EPI": [6608, 6608, 6608],
        "NZE": [1035, 1035, 1035]
    },
    "Total cost (MEUR)": {
        "EPI": [580, 580, 580],
        "NZE": [1083, 1092, 1128]
    }
}

configurations = ['NC', 'DC', 'SC']
colors = ['#4B8BBE', '#306998', '#FFE873']  # Azul, azul oscuro y amarillo pálido

# Función para crear cada gráfico
def plot_parameter_comparison(param_name, epi_vals, nze_vals):
    x = np.arange(len(configurations))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars_epi = ax.bar(x - width/2, epi_vals, width, label='EPI', color='gray', alpha=0.7, edgecolor='black')
    bars_nze = ax.bar(x + width/2, nze_vals, width, label='NZE', color=colors, alpha=1.0, edgecolor='black')

    ax.set_ylabel(param_name, fontsize=12)
    ax.set_title(f'Comparison of {param_name} across Configurations', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(configurations, fontsize=11)
    ax.legend()

    # Añadir etiquetas de valores
    for bar, value in zip(bars_epi, epi_vals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.02 * max(epi_vals + nze_vals),
                f'{value:.2f}', ha='center', va='bottom', fontsize=9)

    # Etiquetas y porcentaje de cambio respecto a NZE-NC
    base_nze = nze_vals[0]
    for i, (bar, value) in enumerate(zip(bars_nze, nze_vals)):
        height = bar.get_height()
        # Valor
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.02 * max(epi_vals + nze_vals),
                f'{value:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        # % de cambio (solo si no es NC)
        if i > 0 and base_nze != 0:
            pct_change = (value - base_nze) / base_nze * 100
            color = 'green' if pct_change > 0 else 'red' if pct_change < 0 else 'gray'
            arrow = '↑' if pct_change > 0 else '↓' if pct_change < 0 else '='
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.09 * max(epi_vals + nze_vals),
                    f'{pct_change:+.1f}% {arrow}', ha='center', va='bottom',
                    fontsize=10, color=color, fontweight='bold')

    ax.set_ylim(0, max(epi_vals + nze_vals) * 1.3)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# Generar gráfico para cada parámetro
for param, values in parameters.items():
    plot_parameter_comparison(param, values["EPI"], values["NZE"])

#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Datos simulados para ilustración
parametros = ['Inertia', 'FFR', 'FFR Gain', 'PFR', 'PFR Gain', 'Spillage', 'Emissions', 'Price', 'VRE Gen', 'VRE Curtailment', 'Penetration', 'Load Shed']

# Valores base (NC) simulados para cada escenario
np.random.seed(0)
NC_values_EPI = np.random.uniform(50, 100, len(parametros))
NC_values_NZE = np.random.uniform(50, 100, len(parametros))

# Porcentajes DC y SC respecto a NC (simulados)
DC_percent_EPI = np.array([48.1, 80.43, 612.39, 87.31, 1733.58, 0.6, 7.6, 110.2, -8.4, 269.2, -6.4, 200])
SC_percent_EPI = np.array([111.7, 123.76, 760.06, 99.35, 2427.81, -3.3, 29.6, 165.0, -21.1, 712.8, -18.0, 300])

DC_percent_NZE = np.array([36.1, 1038.77, 4069.96, 102.13, 1915.92, 28.6, -2.1, 21.8, -2.9, 56.0, -2.2, 100])
SC_percent_NZE = np.array([71.9, 1108.09, 4386.13, 116.86, 2392.74, 15.2, -4.5, 21.8, -6.1, 135.1, -5.7, 100])

# Convertir a valores reales respecto a NC
DC_values_EPI = NC_values_EPI * (1 + DC_percent_EPI / 100)
SC_values_EPI = NC_values_EPI * (1 + SC_percent_EPI / 100)

DC_values_NZE = NC_values_NZE * (1 + DC_percent_NZE / 100)
SC_values_NZE = NC_values_NZE * (1 + SC_percent_NZE / 100)

# Colores elegantes y suaves (diferentes paletas por escenario)
from matplotlib.colors import to_rgba

# Base color por parámetro para EPI (azul grisáceo suave)
base_colors_epi = [
    '#7E9BA2', '#7E9BA2', '#7E9BA2', '#7E9BA2', '#7E9BA2', '#7E9BA2',
    '#7E9BA2', '#7E9BA2', '#7E9BA2', '#7E9BA2', '#7E9BA2', '#7E9BA2'
]
# Base color por parámetro para NZE (verde oliva suave)
base_colors_nze = [
    '#9CA87D', '#9CA87D', '#9CA87D', '#9CA87D', '#9CA87D', '#9CA87D',
    '#9CA87D', '#9CA87D', '#9CA87D', '#9CA87D', '#9CA87D', '#9CA87D'
]

# Crear figura
fig, axes = plt.subplots(4, 3, figsize=(16, 12))
axes = axes.flatten()

bar_width = 0.25
x = np.arange(2)  # 0 for EPI, 1 for NZE

for i, param in enumerate(parametros):
    ax = axes[i]

    # Valores para cada enfoque
    y_nc = [NC_values_EPI[i], NC_values_NZE[i]]
    y_dc = [DC_values_EPI[i], DC_values_NZE[i]]
    y_sc = [SC_values_EPI[i], SC_values_NZE[i]]

    # Posiciones con desplazamiento
    pos_nc = x - bar_width
    pos_dc = x
    pos_sc = x + bar_width

    # Colores degradados por escenario
    color_nc_epi = to_rgba(base_colors_epi[i], 0.4)
    color_dc_epi = to_rgba(base_colors_epi[i], 0.7)
    color_sc_epi = to_rgba(base_colors_epi[i], 1.0)

    color_nc_nze = to_rgba(base_colors_nze[i], 0.4)
    color_dc_nze = to_rgba(base_colors_nze[i], 0.7)
    color_sc_nze = to_rgba(base_colors_nze[i], 1.0)

    # Barras
    ax.bar(pos_nc[0], y_nc[0], width=bar_width, color=color_nc_epi, label='NC' if i == 0 else "", edgecolor='black')
    ax.bar(pos_dc[0], y_dc[0], width=bar_width, color=color_dc_epi, label='DC' if i == 0 else "", edgecolor='black')
    ax.bar(pos_sc[0], y_sc[0], width=bar_width, color=color_sc_epi, label='SC' if i == 0 else "", edgecolor='black')

    ax.bar(pos_nc[1], y_nc[1], width=bar_width, color=color_nc_nze, edgecolor='black')
    ax.bar(pos_dc[1], y_dc[1], width=bar_width, color=color_dc_nze, edgecolor='black')
    ax.bar(pos_sc[1], y_sc[1], width=bar_width, color=color_sc_nze, edgecolor='black')

    # Títulos y ejes
    ax.set_title(param, fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(['EPI', 'NZE'])
    ax.tick_params(axis='x', labelsize=9)
    ax.tick_params(axis='y', labelsize=8)

# Leyenda global
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=12)

fig.suptitle('Comparación por parámetro y escenario (NC, DC, SC)', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


#%%

import matplotlib.pyplot as plt
import numpy as np

# Datos reconstruidos con valores para NC y valores absolutos para todos
parameters = {
    "Inertia (GWs)": {
        "EPI": [48.3, 71.5, 102.3],  # NC, DC, SC
        "NZE": [53.0, 72.1, 91.1]
    },
    "FFR (MW)": {
        "EPI": [60, 80.43, 123.76],
        "NZE": [820, 1038.77, 1108.09]
    },
    "FFR Gain (MW/Hz)": {
        "EPI": [520, 612.39, 760.06],
        "NZE": [3800, 4069.96, 4386.13]
    },
    "PFR (MW)": {
        "EPI": [70, 87.31, 99.35],
        "NZE": [88, 102.13, 116.86]
    },
    "PFR Gain (MW/Hz)": {
        "EPI": [1500, 1733.58, 2427.81],
        "NZE": [1750, 1915.92, 2392.74]
    },
    "Spillage (%)": {
        "EPI": [0.1, 0.1 * 1.006, 0.1 * 0.967],
        "NZE": [1.3, 1.3 * 1.286, 1.3 * 1.152]
    },
    "Emissions (%)": {
        "EPI": [6608, 6608 * 1.076, 6608 * 1.296],
        "NZE": [1035, 1035 * 0.979, 1035 * 0.955]
    },
    "Price (%)": {
        "EPI": [580, 580 * 2.102, 580 * 2.65],
        "NZE": [1083, 1083 * 1.218, 1083 * 1.218]
    },
    "VRE Gen (%)": {
        "EPI": [100, 100 * 0.916, 100 * 0.789],
        "NZE": [100, 100 * 0.971, 100 * 0.939]
    },
    "VRE Curtailment (%)": {
        "EPI": [1, 1 * 3.692, 1 * 8.128],
        "NZE": [2, 2 * 1.56, 2 * 2.351]
    },
    "Penetration (%)": {
        "EPI": [50, 50 * 0.936, 50 * 0.82],
        "NZE": [60, 60 * 0.978, 60 * 0.943]
    },
    "Load Shed (%)": {
        "EPI": [0.1, 0.1 * 3.0, 0.1 * 4.0],
        "NZE": [0.2, 0.2 * 2.0, 0.2 * 2.0]
    }
}

configurations = ['NC', 'DC', 'SC']
scenarios = ['EPI', 'NZE']

# Paleta base de colores elegantes y diferentes para cada parámetro
colors = [
    '#4E79A7', '#F28E2B', '#E15759', '#76B7B2',
    '#59A14F', '#EDC948', '#B07AA1', '#FF9DA7',
    '#9C755F', '#BAB0AC', '#86BCB6', '#D37295'
]

def plot_comparison(param_name, data, color):
    x = np.arange(len(configurations))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))

    # Barras EPI (izquierda del grupo)
    bars_epi = ax.bar(x - width/1.5, data["EPI"], width,
                      color=[color + '80', color + 'b0', color],
                      label='EPI', edgecolor='black')

    # Barras NZE (derecha del grupo)
    bars_nze = ax.bar(x + width/1.5, data["NZE"], width,
                      color=[color + '40', color + '80', color],
                      label='NZE', edgecolor='black', hatch='///')

    # Etiquetas de valores y cambio relativo
    for bars, vals, scenario in zip([bars_epi, bars_nze], [data["EPI"], data["NZE"]], scenarios):
        base = vals[0]
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.03 * max(vals),
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)
            if i > 0 and base != 0:
                pct_change = (vals[i] - base) / base * 100
                arrow = '↑' if pct_change > 0 else '↓' if pct_change < 0 else '='
                color_text = 'green' if pct_change > 0 else 'red' if pct_change < 0 else 'gray'
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.10 * max(vals),
                        f'{pct_change:+.1f}% {arrow}', ha='center', va='bottom',
                        fontsize=10, color=color_text, fontweight='bold')

    ax.set_ylabel(param_name, fontsize=12)
    ax.set_title(f'Comparison of {param_name} across Approaches', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(configurations, fontsize=11)
    ax.legend()
    ax.set_ylim(0, max(data["EPI"] + data["NZE"]) * 1.4)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# Graficar todos los parámetros
for idx, (param, values) in enumerate(parameters.items()):
    base_color = colors[idx % len(colors)]
    plot_comparison(param, values, base_color)
#%%
import matplotlib.pyplot as plt
import numpy as np

# Datos
parameters = {
    "Inertia (GWs)": {
        "EPI": [48.3, 71.5, 102.3],  # NC, DC, SC
        "NZE": [53.0, 72.1, 91.1]
    },
    "FFR (MW)": {
        "EPI": [60, 80.43, 123.76],
        "NZE": [820, 1038.77, 1108.09]
    },
    "FFR Gain (MW/Hz)": {
        "EPI": [520, 612.39, 760.06],
        "NZE": [3800, 4069.96, 4386.13]
    },
    "PFR (MW)": {
        "EPI": [70, 87.31, 99.35],
        "NZE": [88, 102.13, 116.86]
    },
    "PFR Gain (MW/Hz)": {
        "EPI": [1500, 1733.58, 2427.81],
        "NZE": [1750, 1915.92, 2392.74]
    },
    "Spillage (%)": {
        "EPI": [0.1, 0.1 * 1.006, 0.1 * 0.967],
        "NZE": [1.3, 1.3 * 1.286, 1.3 * 1.152]
    },
    "Emissions (%)": {
        "EPI": [6608, 6608 * 1.076, 6608 * 1.296],
        "NZE": [1035, 1035 * 0.979, 1035 * 0.955]
    },
    "Price (%)": {
        "EPI": [580, 580 * 2.102, 580 * 2.65],
        "NZE": [1083, 1083 * 1.218, 1083 * 1.218]
    },
    "VRE Gen (%)": {
        "EPI": [100, 100 * 0.916, 100 * 0.789],
        "NZE": [100, 100 * 0.971, 100 * 0.939]
    },
    "VRE Curtailment (%)": {
        "EPI": [1, 1 * 3.692, 1 * 8.128],
        "NZE": [2, 2 * 1.56, 2 * 2.351]
    },
    "Penetration (%)": {
        "EPI": [50, 50 * 0.936, 50 * 0.82],
        "NZE": [60, 60 * 0.978, 60 * 0.943]
    },
    "Load Shed (%)": {
        "EPI": [0.1, 0.1 * 3.0, 0.1 * 4.0],
        "NZE": [0.2, 0.2 * 2.0, 0.2 * 2.0]
    }
}

configurations = ['NC', 'DC', 'SC']

# Paletas distintas para EPI y NZE (hasta 3 tonos por plot)
epi_palette = ['#e69f00', '#f0c541', '#f7d878']  # cálidos/amarillos
nze_palette = ['#1b9e77', '#66c2a5', '#a6dba0']  # fríos/verdosos

def plot_comparison(param_name, data, epi_colors, nze_colors):
    width = 0.8  # ancho de cada barra
    spacing = 1  # espacio entre EPI y NZE

    # Posiciones
    x_epi = np.arange(3)
    x_nze = x_epi + 3 + spacing  # deja espacio entre los grupos

    fig, ax = plt.subplots(figsize=(10, 6))

    # Barras EPI
    bars_epi = ax.bar(x_epi, data['EPI'], width=width,
                      color=epi_colors, edgecolor='black', label='EPI')

    # Barras NZE
    bars_nze = ax.bar(x_nze, data['NZE'], width=width,
                      color=nze_colors, edgecolor='black', label='NZE')

    # Línea segmentada de separación
    sep_pos = (x_epi[-1] + x_nze[0]) / 2 - spacing / 2
    ax.axvline(x=sep_pos + 0.5, color='gray', linestyle='--', linewidth=1)

    # Etiquetas sobre cada barra
    for bars, vals in zip([bars_epi, bars_nze], [data["EPI"], data["NZE"]]):
        base = vals[0]
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.03 * max(vals),
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)
            if i > 0 and base != 0:
                pct_change = (vals[i] - base) / base * 100
                arrow = '↑' if pct_change > 0 else '↓' if pct_change < 0 else '='
                color_text = 'green' if pct_change > 0 else 'red' if pct_change < 0 else 'gray'
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.1 * max(vals),
                        f'{pct_change:+.1f}% {arrow}', ha='center', va='bottom',
                        fontsize=10, color=color_text, fontweight='bold')

    # Ejes y etiquetas
    ax.set_ylabel(param_name, fontsize=12)
    ax.set_title(f'Comparison of {param_name} across Approaches', fontsize=14)

    x_ticks = list(x_epi) + list(x_nze)
    x_labels = ['NC', 'DC', 'SC'] * 2
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=11)
    ax.legend()
    ax.set_ylim(0, max(data["EPI"] + data["NZE"]) * 1.4)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# Plot para cada parámetro
for idx, (param, values) in enumerate(parameters.items()):
    epi_colors = [epi_palette[i % len(epi_palette)] for i in range(3)]
    nze_colors = [nze_palette[i % len(nze_palette)] for i in range(3)]
    plot_comparison(param, values, epi_colors, nze_colors)

#%%

import matplotlib.pyplot as plt
import numpy as np

# Datos
parameters = {
    "Inertia (GWs)": {
        "EPI": [48.3, 71.5, 102.3],
        "NZE": [53.0, 72.1, 91.1]
    },
    "FFR (MW)": {
        "EPI": [60, 80.43, 123.76],
        "NZE": [820, 1038.77, 1108.09]
    },
    "FFR Gain (MW/Hz)": {
        "EPI": [520, 612.39, 760.06],
        "NZE": [3800, 4069.96, 4386.13]
    },
    "PFR (MW)": {
        "EPI": [70, 87.31, 99.35],
        "NZE": [88, 102.13, 116.86]
    },
    "PFR Gain (MW/Hz)": {
        "EPI": [1500, 1733.58, 2427.81],
        "NZE": [1750, 1915.92, 2392.74]
    },
    "Spillage (%)": {
        "EPI": [0.1, 0.1006, 0.0967],
        "NZE": [1.3, 1.6718, 1.4976]
    },
    "Emissions (%)": {
        "EPI": [6608, 7107.2, 8562.37],
        "NZE": [1035, 1013.685, 987.825]
    },
    "Price (%)": {
        "EPI": [580, 1219.16, 1537],
        "NZE": [1083, 1318.1, 1318.1]
    },
    "VRE Gen (%)": {
        "EPI": [100, 91.6, 78.9],
        "NZE": [100, 97.1, 93.9]
    },
    "VRE Curtailment (%)": {
        "EPI": [1, 3.692, 8.128],
        "NZE": [2, 3.12, 4.702]
    },
    "Penetration (%)": {
        "EPI": [50, 46.8, 41],
        "NZE": [60, 58.68, 56.58]
    },
    "Load Shed (%)": {
        "EPI": [0.1, 0.3, 0.4],
        "NZE": [0.2, 0.4, 0.4]
    }
}

# Paletas de colores únicas para cada plot
epi_palette_list = [
    ['#e69f00', '#f0c541', '#f7d878'],
    ['#d55e00', '#f08080', '#f4a582'],
    ['#a65628', '#dfc27d', '#f6e8c3'],
    ['#b07c6f', '#d4a373', '#e2b07e'],
    ['#c49c94', '#c68642', '#f1c27d'],
    ['#dd8452', '#ffa07a', '#fac898'],
    ['#c67171', '#e79e84', '#f6d5b3'],
    ['#a67b5b', '#f2c18d', '#f6e2b3'],
    ['#c99856', '#e0ac69', '#f6d8a9'],
    ['#cf9f52', '#ddb872', '#f7e6a4'],
    ['#deaa87', '#f0c987', '#f9e7c4'],
    ['#b57f50', '#e1b382', '#f3dabd']
]

nze_palette_list = [
    ['#1b9e77', '#66c2a5', '#a6dba0'],
    ['#377eb8', '#a6cee3', '#b3cde3'],
    ['#4daf4a', '#b2df8a', '#ccebc5'],
    ['#5ab4ac', '#a1d99b', '#c7e9c0'],
    ['#74c476', '#bae4b3', '#e5f5e0'],
    ['#6baed6', '#9ecae1', '#c6dbef'],
    ['#80cdc1', '#a6dba0', '#d9f0d3'],
    ['#41b6c4', '#7fcdbb', '#c7e9b4'],
    ['#66bd63', '#a6d96a', '#d9ef8b'],
    ['#2ca25f', '#99d8c9', '#ccebc5'],
    ['#3288bd', '#66c2a5', '#abdda4'],
    ['#5eacd3', '#9bd5c0', '#d0f0c0']
]

def plot_comparison(param_name, data, epi_colors, nze_colors):
    width = 0.8
    spacing = 1
    x_epi = np.arange(3)
    x_nze = x_epi + 3 + spacing

    fig, ax = plt.subplots(figsize=(10, 6))
    bars_epi = ax.bar(x_epi, data['EPI'], width=width, color=epi_colors, edgecolor='black', label='EPI')
    bars_nze = ax.bar(x_nze, data['NZE'], width=width, color=nze_colors, edgecolor='black', label='NZE')

    sep_pos = (x_epi[-1] + x_nze[0]) / 2 - spacing / 2
    ax.axvline(x=sep_pos + 0.5, color='gray', linestyle='--', linewidth=1)

    for bars, vals in zip([bars_epi, bars_nze], [data["EPI"], data["NZE"]]):
        base = vals[0]
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.03 * max(vals),
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)
            if i > 0 and base != 0:
                pct_change = (vals[i] - base) / base * 100
                arrow = '↑' if pct_change > 0 else '↓' if pct_change < 0 else '='
                color_text = 'green' if pct_change > 0 else 'red' if pct_change < 0 else 'gray'
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.1 * max(vals),
                        f'{pct_change:+.1f}% {arrow}', ha='center', va='bottom',
                        fontsize=10, color=color_text, fontweight='bold')

    ax.set_ylabel(param_name, fontsize=12)
    ax.set_title(f'Comparison of {param_name} across Approaches', fontsize=14)
    ax.set_xticks(list(x_epi) + list(x_nze))
    ax.set_xticklabels(['NC', 'DC', 'SC'] * 2, fontsize=11)
    ax.legend()
    ax.set_ylim(0, max(data["EPI"] + data["NZE"]) * 1.4)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# Graficar todos los parámetros con paletas distintas
for idx, (param, values) in enumerate(parameters.items()):
    epi_colors = epi_palette_list[idx % len(epi_palette_list)]
    nze_colors = nze_palette_list[idx % len(nze_palette_list)]
    plot_comparison(param, values, epi_colors, nze_colors)

#%%


import matplotlib.pyplot as plt
import numpy as np

# Datos
parameters = {
    "Inertia (GWs)": {
        "EPI": [48.3, 71.5, 102.3],
        "NZE": [53.0, 72.1, 91.1]
    },
    "FFR (MW)": {
        "EPI": [60, 80.43, 123.76],
        "NZE": [820, 1038.77, 1108.09]
    },
    "FFR Gain (MW/Hz)": {
        "EPI": [520, 612.39, 760.06],
        "NZE": [3800, 4069.96, 4386.13]
    },
    "PFR (MW)": {
        "EPI": [70, 87.31, 99.35],
        "NZE": [88, 102.13, 116.86]
    },
    "PFR Gain (MW/Hz)": {
        "EPI": [1500, 1733.58, 2427.81],
        "NZE": [1750, 1915.92, 2392.74]
    },
    "Spillage (%)": {
        "EPI": [0.1, 0.1006, 0.0967],
        "NZE": [1.3, 1.6718, 1.4976]
    },
    "Emissions (%)": {
        "EPI": [6608, 7107.2, 8562.37],
        "NZE": [1035, 1013.685, 987.825]
    },
    "Price (%)": {
        "EPI": [580, 1219.16, 1537],
        "NZE": [1083, 1318.1, 1318.1]
    },
    "VRE Gen (%)": {
        "EPI": [100, 91.6, 78.9],
        "NZE": [100, 97.1, 93.9]
    },
    "VRE Curtailment (%)": {
        "EPI": [1, 3.692, 8.128],
        "NZE": [2, 3.12, 4.702]
    },
    "Penetration (%)": {
        "EPI": [50, 46.8, 41],
        "NZE": [60, 58.68, 56.58]
    },
    "Load Shed (%)": {
        "EPI": [0.1, 0.3, 0.4],
        "NZE": [0.2, 0.4, 0.4]
    }
}

from matplotlib.colors import to_rgb
import seaborn as sns

def adjust_brightness(color, factor):
    r, g, b = to_rgb(color)
    return (min(r * factor, 1), min(g * factor, 1), min(b * factor, 1))

# Paletas más vivas por variable (12 subplots)
base_epi_colors = [
    "#2ca02c", "#ff7f0e", "#d62728", "#9467bd",
    "#8c564b", "#7f7f7f", "#bcbd22", "#e377c2",
    "#17becf", "#6a3d9a", "#b15928", "#e31a1c"
]

base_nze_colors = [
    "#1f77b4", "#17becf", "#8c564b", "#393b79",
    "#e7969c", "#1b9e77", "#e7298a", "#a6cee3",
    "#fb9a99", "#377eb8", "#66c2a5", "#6baed6"
]

# Usamos niveles más intensos: 0.8, 1.0, 1.2
epi_palette_list = [[
    adjust_brightness(color, 0.8),
    adjust_brightness(color, 1.0),
    adjust_brightness(color, 1.2)
] for color in base_epi_colors]

nze_palette_list = [[
    adjust_brightness(color, 0.8),
    adjust_brightness(color, 1.0),
    adjust_brightness(color, 1.2)
] for color in base_nze_colors]




def plot_comparison(ax, param_name, data, epi_colors, nze_colors):
    width = 0.8
    spacing = 1
    x_epi = np.arange(3)
    x_nze = x_epi + 3 + spacing

    bars_epi = ax.bar(x_epi, data['EPI'], width=width, color=epi_colors, edgecolor='black', label='EPI')
    bars_nze = ax.bar(x_nze, data['NZE'], width=width, color=nze_colors, edgecolor='black', label='NZE')

    sep_pos = (x_epi[-1] + x_nze[0]) / 2 - spacing / 2
    ax.axvline(x=sep_pos + 0.5, color='gray', linestyle='--', linewidth=1)

    for bars, vals in zip([bars_epi, bars_nze], [data["EPI"], data["NZE"]]):
        base = vals[0]
        for bar, val in zip(bars[1:], vals[1:]):
            delta = (val - base) / base * 100
            color = 'green' if delta >= 0 else 'red'
            arrow = '↑' if delta >= 0 else '↓'
            ax.annotate(f"{arrow}{abs(delta):.1f}%", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 5), textcoords='offset points', ha='center', color=color, fontsize=9)
    ax.set_title(param_name, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])

# Crear la grilla de subplots
fig, axs = plt.subplots(4, 3, figsize=(20, 16))
axs = axs.flatten()

for i, (param, data) in enumerate(parameters.items()):
    epi_colors = epi_palette_list[i]
    nze_colors = nze_palette_list[i]
    plot_comparison(axs[i], param, data, epi_colors, nze_colors)

# Ajuste de layout
plt.tight_layout()
plt.show()


