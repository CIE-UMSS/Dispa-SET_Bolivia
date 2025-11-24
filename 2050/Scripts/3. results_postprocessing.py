# -*- coding: utf-8 -*-
"""
Created on Tue May 20 15:33:53 2025

@author: navia
"""
#%% PLOTS INDIVIDUALES AVERAGE METRICS ACROSS SCENARIOS AND APPROACHES
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# Datos 
parameters = {
    "System Inertia (GWs)": {
        "EPI": [13.35, 19.78, 28.26],
        "NZE": [16.27, 22.15, 27.96]
    },
    "FFR (MW)": {
        "EPI": [0, 80.43, 123.76],
        "NZE": [0, 1038.77, 1108.09]
    },
    "PFR (MW)": {
        "EPI": [0, 87.31, 99.35],
        "NZE": [0, 102.13, 116.86]
    },
    "Spillage (MWh)": {
        "EPI": [41.29, 41.52, 39.92],
        "NZE": [52.73, 67.84, 60.74]
    },
    "Emissions (kgCO2,eq/MWh)": {
        "EPI": [189.51, 203.96, 245.63],
        "NZE": [122.07, 119.51, 116.56]
    },
    "Shadow Price (EUR/MWh)": {
        "EPI": [9.97, 6.48, 4.92],
        "NZE": [1.96, 0.23, 0.11]
    },
    "Electricity Price (EUR/MWh)": {
        "EPI": [10.03, 6.51, 5.09],
        "NZE": [1.91, 0.23, 0.12]
    },
    "Total Generation (TWh)": {
        "EPI": [50.59, 49.53, 48.74],
        "NZE": [104.11, 103.46, 103.64]
    },
    "Total VRE Generation (TWh)": {
        "EPI": [27.45, 25.14, 21.67],
        "NZE": [68.11, 66.16, 63.94]
    },
    "Penetration VRE (%)": {
        "EPI": [54.26, 50.76, 44.46],
        "NZE": [65.42, 63.95, 61.69]
    },
    "Total VRE Curtailment (TWh)": {
        "EPI": [0.78, 2.88, 6.34],
        "NZE": [2.93, 4.57, 6.89]
    },
    "Curtailment VRE (%)": {
        "EPI": [2.76, 10.28, 22.63],
        "NZE": [4.12, 6.46, 9.73]
    },
    "Load Sheeding (TWh)": {
        "EPI": [0.01, 0.03, 0.04],
        "NZE": [0.01, 0.02, 0.02]
    },
}

# Configuración general
categories = ['NC', 'DC', 'SC']
scenarios = list(next(iter(parameters.values())).keys())
num_scenarios = len(scenarios)
num_categories = len(categories)
bar_colors = [
    "#A8DADC",  # soft aqua
    "#FFBCBC",  # soft red
    # "#FFE3A9",  # light yellow
    # "#B5EAD7",  # mint
    # "#CBAACB",  # soft purple
    # "#FFD6A5",  # peach
]
trend_colors = ['#1f78b4', '#ff7f0e', '#2ca02c']

# Función para graficar
def plot_grouped_with_trends(param_name, data, bar_colors, trend_colors):
    fig, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.12
    index = np.arange(num_categories)

    values_by_scenario = [data[sc] for sc in scenarios]
    values_by_category = list(zip(*values_by_scenario))

    for i, scenario in enumerate(scenarios):
        x = index + (i - num_scenarios / 2) * bar_width + bar_width / 2
        values = data[scenario]
        ax.bar(
            x, values, bar_width,
            color=bar_colors[i], edgecolor='gray',
            alpha=0.5, linewidth=0.6
        )
        for j, val in enumerate(values):
            ax.text(x[j], val + 0.02 * max(values), f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    for i, cat_values in enumerate(values_by_category):
        x_trend = index[i] + (np.arange(num_scenarios) - num_scenarios / 2) * bar_width + bar_width / 2
        ax.plot(
            x_trend, cat_values, marker='o', linestyle='-',
            linewidth=1.5, color=trend_colors[i],
            label=f'Trend {categories[i]}'
        )

    ax.set_ylabel(param_name, fontsize=12)
    ax.set_title(f'{param_name} across Scenarios and Approaches', fontsize=14, weight='bold')
    ax.set_xticks(index)
    ax.set_xticklabels(categories, fontsize=11)
    ax.tick_params(axis='y', labelsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    legend_elements = [
        Line2D([0], [0], color=bar_colors[i], lw=6, label=scenarios[i]) for i in range(num_scenarios)
    ] + [
        Line2D([0], [0], color=trend_colors[i], lw=2, marker='o', label=f'Trend {categories[i]}') for i in range(num_categories)
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc='upper left', frameon=True)

    plt.tight_layout()
    plt.show()

# Iterar sobre todos los parámetros
for param, data in parameters.items():
    plot_grouped_with_trends(param, data, bar_colors, trend_colors)

#%% GRILLA DE PLOTS AVERAGE METRICS ACROSS SCENARIOS AND APPROACHES

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# Datos 
parameters = {
    "System Inertia (GWs)": {
        "EPI": [13.35, 19.78, 28.26],
        "NZE": [16.27, 22.15, 27.96]
    },
    "FFR (MW)": {
        "EPI": [0, 80.43, 123.76],
        "NZE": [0, 1038.77, 1108.09]
    },
    "PFR (MW)": {
        "EPI": [0, 87.31, 99.35],
        "NZE": [0, 102.13, 116.86]
    },
    "Spillage (MWh)": {
        "EPI": [41.29, 41.52, 39.92],
        "NZE": [52.73, 67.84, 60.74]
    },
    "Emissions (kgCO2,eq/MWh)": {
        "EPI": [189.51, 203.96, 245.63],
        "NZE": [122.07, 119.51, 116.56]
    },
    "Shadow Price (EUR/MWh)": {
        "EPI": [9.97, 6.48, 4.92],
        "NZE": [1.96, 0.23, 0.11]
    },
    "Electricity Price (EUR/MWh)": {
        "EPI": [10.03, 6.51, 5.09],
        "NZE": [1.91, 0.23, 0.12]
    },
    "Total Generation (TWh)": {
        "EPI": [50.59, 49.53, 48.74],
        "NZE": [104.11, 103.46, 103.64]
    },
    "Total VRE Generation (TWh)": {
        "EPI": [27.45, 25.14, 21.67],
        "NZE": [68.11, 66.16, 63.94]
    },
    "Penetration VRE (%)": {
        "EPI": [54.26, 50.76, 44.46],
        "NZE": [65.42, 63.95, 61.69]
    },
    "Total VRE Curtailment (TWh)": {
        "EPI": [0.78, 2.88, 6.34],
        "NZE": [2.93, 4.57, 6.89]
    },
    "Curtailment VRE (%)": {
        "EPI": [2.76, 10.28, 22.63],
        "NZE": [4.12, 6.46, 9.73]
    },
    "Load Sheeding (TWh)": {
        "EPI": [0.01, 0.03, 0.04],
        "NZE": [0.01, 0.02, 0.02]
    },
}


# Configuración
categories = ['NC', 'DC', 'SC']
scenarios = list(next(iter(parameters.values())).keys())
bar_colors = [
    "#A8DADC",  # soft aqua
    "#FFBCBC",  # soft red
    # "#FFE3A9",  # light yellow
    # "#B5EAD7",  # mint
    # "#CBAACB",  # soft purple
    # "#FFD6A5",  # peach
]
trend_colors = ['#1f78b4', '#ff7f0e', '#2ca02c']

def plot_subplot(ax, param_name, data, bar_colors, trend_colors):
    bar_width = 0.12
    index = np.arange(len(categories))
    values_by_scenario = [data[sc] for sc in scenarios]
    values_by_category = list(zip(*values_by_scenario))

    for i, scenario in enumerate(scenarios):
        x = index + (i - len(scenarios)/2) * bar_width + bar_width/2
        values = data[scenario]
        ax.bar(x, values, bar_width, color=bar_colors[i], edgecolor='gray', alpha=0.5, linewidth=0.6)

    for i, cat_values in enumerate(values_by_category):
        x_trend = index[i] + (np.arange(len(scenarios)) - len(scenarios)/2) * bar_width + bar_width/2
        ax.plot(x_trend, cat_values, marker='.', linestyle='-.', linewidth=1, color=trend_colors[i])

    ax.set_title(param_name, fontsize=15)
    ax.set_xticks(index)
    ax.set_xticklabels(categories, fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

# Crear la grilla de subplots
n = len(parameters)
cols = 3
rows = int(np.ceil(n / cols))

fig, axes = plt.subplots(rows, cols, figsize=(cols*6, rows*4))
axes = axes.flatten()

for idx, (param, data) in enumerate(parameters.items()):
    plot_subplot(axes[idx], param, data, bar_colors, trend_colors)

# Eliminar ejes vacíos
for ax in axes[n:]:
    fig.delaxes(ax)

# Leyenda general
handles = [
    Line2D([0], [0], color=bar_colors[i], lw=6, label=scenarios[i]) for i in range(len(scenarios))
] + [
    Line2D([0], [0], color=trend_colors[i], lw=2, marker='.', label=f'Trend {categories[i]}') for i in range(len(categories))
]
fig.legend(
    handles=handles,
    loc='lower right',
    bbox_to_anchor=(0.8, 0.05),  # mueve la leyenda un poco a la izquierda y arriba
    ncol=1,
    fontsize=15,
    frameon=True
)
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.suptitle("Average Metrics Across Scenarios and Approaches", fontsize=15, weight='bold')
plt.show()

#%%GRILLA DE PLOTS AVERAGE METRICS ACROSS SCENARIOS AND APPROACHES
#  FOR THE PAPER

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# Datos 
parameters = {
    "System Inertia (GWs)": {
        "EPI": [13.35, 19.78, 28.26],
        "NZE": [16.27, 22.15, 27.96]
    },
    "FFR (MW)": {
        "EPI": [0, 321.71, 495.05],
        "NZE": [0, 1033.08, 1108.09]
    },
    "PFR (MW)": {
        "EPI": [0, 87.31, 99.35],
        "NZE": [0, 102.13, 116.86]
    },
    "Spillage (MWh)": {
        "EPI": [41.29, 41.52, 39.92],
        "NZE": [52.73, 67.84, 60.74]
    },
    "Emissions (kgCO2,eq/MWh)": {
        "EPI": [189.51, 203.96, 245.63],
        "NZE": [122.07, 119.51, 116.56]
    },
    "Shadow Price (EUR/MWh)": {
        "EPI": [303.09, 637.52, 803.27],
        "NZE": [115.09, 140.22, 140.24]
    },
    "Penetration VRE (%)": {
        "EPI": [54.26, 50.76, 44.46],
        "NZE": [65.42, 63.95, 61.69]
    },
    "Curtailment VRE (%)": {
        "EPI": [2.76, 10.28, 22.63],
        "NZE": [4.12, 6.46, 9.73]
    },
}

# Configuración
categories = ['NC', 'DC', 'SC']
scenarios = list(next(iter(parameters.values())).keys())
bar_colors = [
    "#3B4A67",  # EPI
    "#6CCE71",  # NZE (verde)
]
trend_colors = ['#6E2D9B', '#485C42']  # línea para EPI, NZE

def plot_subplot(ax, param_name, data, bar_colors, trend_colors):
    bar_width = 0.4
    index = np.arange(len(categories))  # NC, DC, SC
    ymax = 0

    for i, scenario in enumerate(scenarios):  # EPI y NZE
        x = index + (i - len(scenarios)/2) * bar_width + bar_width/2
        values = data[scenario]
        ax.bar(x, values, bar_width, color=bar_colors[i], edgecolor='gray', alpha=0.6, linewidth=0.6)

        ref_value = values[0]
        for j in range(len(values)):
            xi = x[j]
            val = values[j]
            # Valor numérico exacto
            val_txt = f"{val:.1f}"
            text_y = val + max(values)*0.02
            ax.text(xi, text_y, val_txt, ha='center', va='bottom', fontsize=14, weight='bold', color='black')

            if j == 0:
                continue

            # Porcentaje respecto a la barra NC (base)
            base_pct = 0 if ref_value == 0 else 100 * (val - ref_value) / ref_value
            if abs(base_pct) < 1e-2:
                sym_base, color_base = '=', 'gray'
            elif base_pct > 0:
                sym_base, color_base = '↑', 'green'
            else:
                sym_base, color_base = '↓', 'red'

            txt_base = f"{sym_base} {abs(base_pct):.0f}%"
            ax.text(xi, text_y + max(values)*0.06, txt_base, ha='center', va='bottom', fontsize=13, color=color_base)

            # Porcentaje respecto a barra anterior del mismo enfoque
            prev_value = values[j - 1]
            step_pct = 0 if prev_value == 0 else 100 * (val - prev_value) / prev_value
            if abs(step_pct) < 1e-2:
                sym_step, color_step = '=', 'gray'
            elif step_pct > 0:
                sym_step, color_step = '↑', 'blue'
            else:
                sym_step, color_step = '↓', 'orange'

            txt_step = f"{sym_step} {abs(step_pct):.0f}%"
            ax.text(xi, text_y + max(values)*0.12, txt_step, ha='center', va='bottom', fontsize=12, color=color_step)

            ymax = max(ymax, text_y + max(values)*0.12)

    # Líneas de tendencia
    for i, scenario in enumerate(scenarios):
        values = data[scenario]
        x = index + (i - len(scenarios)/2) * bar_width + bar_width/2
        ax.plot(x, values, marker='o', linestyle='--', linewidth=1.8, color=trend_colors[i], label=scenario)
        ymax = max(ymax, max(values))

    ax.set_ylim(0, ymax * 1.20)
    ax.set_title(param_name, fontsize=20)
    ax.set_xticks(index)
    ax.set_xticklabels(categories, fontsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.grid(axis='y', linestyle='--', alpha=0.5)


# Crear grilla de subplots
n = len(parameters)
cols = 2
rows = int(np.ceil(n / cols))

fig, axes = plt.subplots(rows, cols, figsize=(cols*10, rows*4.5))
axes = axes.flatten()

for idx, (param, data) in enumerate(parameters.items()):
    plot_subplot(axes[idx], param, data, bar_colors, trend_colors)

# Eliminar ejes vacíos
for ax in axes[n:]:
    fig.delaxes(ax)

# Leyenda
handles = [
    Line2D([0], [0], color=bar_colors[i], lw=6, label=scenarios[i]) for i in range(len(scenarios))
] + [
    Line2D([0], [0], color=trend_colors[i], lw=2, marker='o', label=f'Trend {scenarios[i]}') for i in range(len(scenarios))
]

fig.legend(
    handles=handles,
    loc='center left',
    bbox_to_anchor=(0.88, 0.88),
    ncol=1,
    fontsize=18,
    frameon=True
)

plt.tight_layout(rect=[0, 0, 0.88, 0.96])
fig.suptitle("Average Metrics Across Scenarios and Approaches", fontsize=25, weight='bold')
plt.show()



#%% PRESENTACION ECOS GRILLA DE PLOTS AVERAGE METRICS ACROSS SCENARIOS AND APPROACHES (SEPARADO EN 2 FIGURAS)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# Datos
parameters = {
    "System Inertia (GWs)": {
        "EPI": [13.35, 19.78, 27.96],
        "NZE": [16.27, 22.15, 28.26]
    },
    "FFR (MW)": {
        "EPI": [0, 321.71, 495.05],
        "NZE": [0, 1033.08, 1108.09]
    },
    "PFR (MW)": {
        "EPI": [0, 87.31, 99.35],
        "NZE": [0, 102.13, 116.86]
    },
    "Spillage (MWh)": {
        "EPI": [41.29, 41.52, 39.92],
        "NZE": [52.73, 67.84, 60.74]
    },
    "Emissions (kgCO2,eq/MWh)": {
        "EPI": [189.51, 203.96, 245.63],
        "NZE": [116.56, 119.51, 122.07]
    },
    "Shadow Price (EUR/MWh)": {
        "EPI": [303.09, 637.52, 803.27],
        "NZE": [115.09, 140.22, 140.24]
    },
    "Penetration VRE (%)": {
        "EPI": [54.26, 50.76, 44.46],
        "NZE": [65.42, 63.95, 61.69]
    },
    "Curtailment VRE (%)": {
        "EPI": [2.76, 10.28, 22.63],
        "NZE": [4.12, 6.46, 9.73]
    },
}

# Configuración
categories = ['NC', 'DC', 'SC']
scenarios = list(next(iter(parameters.values())).keys())  # ['EPI', 'NZE']

# Colores profesionales (ejemplo: tonos azules y verdes)
bar_colors = ["#2E86AB", "#58A55C"]  # EPI (azul), NZE (verde)
trend_colors = ['#155D8B', '#3E7D44']  # tonos más oscuros para líneas de tendencia

# Nuevos colores para líneas de tendencia por categoría en la segunda figura
trend_colors_categories = {
    'NC': 'black',  # negro
    'DC': '#2980B9',  # azul
    'SC': '#27AE60',  # verde
}

# Función de subplot con líneas de tendencia y diferencias internas (sin cambios)
def plot_subplot_trends(ax, param_name, data, bar_colors, trend_colors):
    bar_width = 0.4
    index = np.arange(len(categories))

    # Definir escala y max valor
    max_val = max(max(data[sc]) for sc in scenarios)
    ymax = max_val * 1.2 if max_val > 0 else 1

    for i, scenario in enumerate(scenarios):
        x = index + (i - len(scenarios)/2) * bar_width + bar_width/2
        values = data[scenario]
        ax.bar(x, values, bar_width, color=bar_colors[i], edgecolor='gray', alpha=0.7, linewidth=0.6)

        for j in range(len(values)):
            xi = x[j]
            val = values[j]

            spacing_pct = ymax * 0.05  # espacio para % arriba
            spacing_val = ymax * 0.12  # espacio para valor

            if j > 0:
                prev_value = values[j - 1]
                if prev_value == 0:
                    step_pct = 0
                else:
                    step_pct = 100 * (val - prev_value) / prev_value

                if abs(step_pct) < 1e-2:
                    sym_step, color_step = '=', 'gray'
                elif step_pct > 0:
                    sym_step, color_step = '↑', 'green'  # azul
                else:
                    sym_step, color_step = '↓', 'red'  # naranja

                y_pct = val + spacing_pct
                ax.text(xi, y_pct, f"{sym_step} {abs(step_pct):.0f}%", ha='center', va='bottom',
                        fontsize=14, color=color_step)

            # Valores en negro (negrita y tamaño 14)
            y_val = val + spacing_val
            ax.text(xi, y_val, f"{val:.1f}", ha='center', va='bottom',
                    fontsize=14, weight='bold', color='black')

    for i, scenario in enumerate(scenarios):
        values = data[scenario]
        x = index + (i - len(scenarios)/2) * bar_width + bar_width/2
        ax.plot(x, values, marker='.', linestyle='-.', linewidth=1.5, color=trend_colors[i], label=scenario)

    ax.set_ylim(0, ymax)
    ax.set_title(param_name, fontsize=20)
    ax.set_xticks(index)
    ax.set_xticklabels(categories, fontsize=18, rotation=0)
    ax.tick_params(axis='y', labelsize=18)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

# Subplot solo diferencias % NZE vs EPI, con líneas de tendencia para categorías (modificado)
def plot_subplot_diffs_only(ax, param_name, data, bar_colors):
    bar_width = 0.4
    index = np.arange(len(categories))

    max_val = max(max(data[sc]) for sc in scenarios)
    ymax = max_val * 1.2 if max_val > 0 else 1

    for i, scenario in enumerate(scenarios):
        x = index + (i - len(scenarios)/2) * bar_width + bar_width/2
        values = data[scenario]
        ax.bar(x, values, bar_width, color=bar_colors[i], edgecolor='gray', alpha=0.7, linewidth=0.6)

        for j in range(len(values)):
            xi = x[j]
            val = values[j]

            spacing_val = ymax * 0.10
            y_val = val + spacing_val
            ax.text(xi, y_val, f"{val:.1f}", ha='center', va='bottom',
                    fontsize=14, weight='bold', color='black')

    # Añadir líneas de diferencia con flechas arriba
    for j in range(len(categories)):
        val_epi = data['EPI'][j]
        val_nze = data['NZE'][j]

        if val_epi == 0:
            diff_pct = 0
            sym, color = '=', 'gray'
        else:
            diff_pct = 100 * (val_nze - val_epi) / val_epi
            if abs(diff_pct) < 1e-2:
                sym, color = '=', 'gray'
            elif diff_pct > 0:
                sym, color = '↑', 'green'  # púrpura
            else:
                sym, color = '↓', 'red'

        spacing_pct = ymax * 0.05
        x_center = index[j]
        y_center = max(val_epi, val_nze) + spacing_pct

        ax.text(x_center, y_center, f"{sym} {abs(diff_pct):.0f}%", ha='center', va='bottom',
                fontsize=14, color=color)

        # Línea de diferencia entre barras EPI y NZE
        x_epi = index[j] - bar_width/2
        x_nze = index[j] + bar_width/2
        y_epi = val_epi
        y_nze = val_nze
        ax.plot([x_epi, x_nze], [y_epi, y_nze], color='#8E44AD', linestyle='-.', linewidth=1.5)

    # NUEVO: Añadir líneas de tendencia para cada categoría a través de las 2 barras (EPI y NZE)
    for cat_idx, cat in enumerate(categories):
        x_line = [index[cat_idx] - bar_width/2, index[cat_idx] + bar_width/2]
        y_line = [data['EPI'][cat_idx], data['NZE'][cat_idx]]
        ax.plot(x_line, y_line, color=trend_colors_categories[cat], marker='.', linestyle='-.', linewidth=1.5, label=f'Trend {cat}')

    ax.set_ylim(0, ymax)
    ax.set_title(param_name, fontsize=20)
    ax.set_xticks(index)
    ax.set_xticklabels(categories, fontsize=18, rotation=0)
    ax.tick_params(axis='y', labelsize=18)
    ax.grid(axis='y', linestyle='--', alpha=0.5)


# Crear primera figura (líneas de tendencia)
n = len(parameters)
cols = 2
rows = int(np.ceil(n / cols))

fig1, axes1 = plt.subplots(rows, cols, figsize=(cols * 10, rows * 4))
axes1 = axes1.flatten()

for idx, (param, data) in enumerate(parameters.items()):
    plot_subplot_trends(axes1[idx], param, data, bar_colors, trend_colors)

for ax in axes1[n:]:
    fig1.delaxes(ax)

handles = [
    Line2D([0], [0], color=bar_colors[i], lw=6, label=scenarios[i]) for i in range(len(scenarios))
] + [
    Line2D([0], [0], color=trend_colors[i], lw=2, marker='.', label=f'Trend {scenarios[i]}') for i in range(len(scenarios))
]

fig1.legend(
    handles=handles,
    loc='center left',
    bbox_to_anchor=(0.87, 0.88),
    ncol=1,
    fontsize=20,
    frameon=True
)

plt.tight_layout(rect=[0, 0, 0.88, 0.96])
fig1.suptitle("Average UCED Metrics by Scenario with Trend Lines Comparing Approaches", fontsize=25, weight='bold')

# Crear segunda figura (solo diferencias % NZE vs EPI)
fig2, axes2 = plt.subplots(rows, cols, figsize=(cols * 10, rows * 4))
axes2 = axes2.flatten()

for idx, (param, data) in enumerate(parameters.items()):
    plot_subplot_diffs_only(axes2[idx], param, data, bar_colors)

for ax in axes2[n:]:
    fig2.delaxes(ax)

handles2 = [
    Line2D([0], [0], color=bar_colors[i], lw=6, label=scenarios[i]) for i in range(len(scenarios))
] + [
    Line2D([0], [0], color=color, lw=3, label=f'Trend {cat}') for cat, color in trend_colors_categories.items()
]

fig2.legend(
    handles=handles2,
    loc='center left',
    bbox_to_anchor=(0.87, 0.88),
    ncol=1,
    fontsize=20,
    frameon=True
)

plt.tight_layout(rect=[0, 0, 0.88, 0.96])
fig2.suptitle("Average UCED Metrics by Approach with Trend Lines Comparing Scenarios", fontsize=25, weight='bold')

plt.show()
