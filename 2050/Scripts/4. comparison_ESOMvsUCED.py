# -*- coding: utf-8 -*-
"""
Created on Mon May 26 09:25:32 2025

@author: navia
"""

#%% COMPARISON BETWEEN ESOM AND UCED FOR DIFFERENT APPROACHES (SCENARIO EPI)
import matplotlib.pyplot as plt
import numpy as np

# Tecnologías y colores
technologies = ['Hydro', 'Thermal', 'Wind', 'Solar', 'Biomass', 'Geothermal', 'Batteries', 'Curtailment']
colors = {
    'Hydro': '#0099cc',
    'Thermal': '#e07b00',
    'Wind': '#66cc66',
    'Solar': '#ffcc00',
    'Biomass': '#ff9966',
    'Geothermal': '#8B4513',
    'Batteries': '#4B4B4B',
    'Curtailment': 'red'
}

# Datos por escenario
epi_esom_nc = [9.27, 0.045, 7.59, 19.73, 3.22, 8.17, 2.21, 5.87]
epi_uced_nc = [11.46, 1.98, 8.37, 19.09, 1.22, 6.84, 1.64, 0.78]
epi_uced_dc = [11.32, 2.22, 7.51, 17.63, 3.31, 6.68, 0.88, 2.88]
epi_uced_sc = [10.91, 3.13, 6.18, 15.48, 6.73, 5.74, 0.57, 6.34]

# Porcentajes entre paréntesis
porcentajes = {
    'UCED NC': ['↑23.6%', '↑4299%', '↑10.3%', '↓3.2%', '↓62.1%', '↓16.3%', '↓25.8%', '↓86.7%'],
    'UCED DC': ['↑22.1%', '↑4833%', '↓1.1%', '↓10.7%', '↑2.8%', '↓18.2%', '↓60.2%', '↓50.9%'],
    'UCED SC': ['↑17.7%', '↑6844%', '↓18.6%', '↓21.5%', '↑109%', '↓29.7%', '↓74.2%', '↑8.0%']
}

group1 = ['Hydro', 'Thermal', 'Wind', 'Solar']
group2 = ['Biomass', 'Geothermal', 'Batteries', 'Curtailment']
group1_idx = [technologies.index(t) for t in group1]
group2_idx = [technologies.index(t) for t in group2]

def extract_group(data, group_indices):
    return [data[i] for i in group_indices]

# Offset de texto para cada subplot
offsets = [4, 2.5]

fig, axes = plt.subplots(2, 1, figsize=(30, 10), sharex=False)
width = 0.23

def plot_group(ax, group, idxs, idx):
    x = np.arange(len(group))

    vals_esom = extract_group(epi_esom_nc, idxs)
    vals_uced_nc = extract_group(epi_uced_nc, idxs)
    vals_uced_dc = extract_group(epi_uced_dc, idxs)
    vals_uced_sc = extract_group(epi_uced_sc, idxs)

    bars1 = ax.bar(x - 1.5*width, vals_esom, width, label='EPI ESOM NC',
                   color=[colors[tech] for tech in group], alpha=0.4, edgecolor='black')
    bars2 = ax.bar(x - 0.5*width, vals_uced_nc, width, label='EPI UCED NC',
                   color=[colors[tech] for tech in group], alpha=0.6, edgecolor='black')
    bars3 = ax.bar(x + 0.5*width, vals_uced_dc, width, label='EPI UCED DC',
                   color=[colors[tech] for tech in group], alpha=0.8, edgecolor='black')
    bars4 = ax.bar(x + 1.5*width, vals_uced_sc, width, label='EPI UCED SC',
                   color=[colors[tech] for tech in group], alpha=1.0, edgecolor='black')

    ax.set_ylabel('Dispatch [TWh]', fontsize=20)
    ax.set_xticks(x)
    ax.set_xticklabels(group, fontsize=20)
    ax.tick_params(axis='y', labelsize=18)

    for i in range(len(group) - 1):
        ax.axvline(x=i + 0.5, color='gray', linestyle='--', linewidth=1)

    def add_labels(bars, values, label, porcentajes_local):
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                    f'{label}\n{val:.2f}', ha='center', va='bottom', fontsize=15, fontweight='bold')

            if label != 'ESOM' and porcentajes_local:
                perc = porcentajes_local[i]
                color = 'green' if '↑' in perc else 'red'
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offsets[idx],
                        perc, ha='center', va='bottom', fontsize=15, color=color)

    add_labels(bars1, vals_esom, "ESOM", None)
    add_labels(bars2, vals_uced_nc, "UCED NC", extract_group(porcentajes['UCED NC'], idxs))
    add_labels(bars3, vals_uced_dc, "UCED DC", extract_group(porcentajes['UCED DC'], idxs))
    add_labels(bars4, vals_uced_sc, "UCED SC", extract_group(porcentajes['UCED SC'], idxs))

    all_vals = vals_esom + vals_uced_nc + vals_uced_dc + vals_uced_sc
    ax.set_ylim(0, max(all_vals) + 8)

# Plot para cada grupo
plot_group(axes[0], group1, group1_idx, 0)
axes[0].set_title('ESCENARIO EPI: Dispatch Comparison by Technology (Group 1: Hydro - Solar)', fontsize=25)

plot_group(axes[1], group2, group2_idx, 1)
axes[1].set_title('ESCENARIO EPI: Dispatch Comparison by Technology (Group 2: Biomass - Curtailment)', fontsize=25)

plt.tight_layout()
plt.show()


#%% COMPARISON BETWEEN ESOM AND UCED FOR DIFFERENT APPROACHES (SCENARIO NZE)
import matplotlib.pyplot as plt
import numpy as np

# Tecnologías y colores
technologies = ['Hydro', 'Thermal', 'Wind', 'Solar', 'Biomass', 'Geothermal', 'Batteries', 'Curtailment']
colors = {
    'Hydro': '#0099cc', 'Thermal': '#e07b00', 'Wind': '#66cc66', 'Solar': '#ffcc00',
    'Biomass': '#ff9966', 'Geothermal': '#8B4513', 'Batteries': '#4B4B4B', 'Curtailment': 'red'
}

# Datos por escenario
nze_esom_nc = [9.14, 0.0009, 16.89, 53.97, 5.91, 7.82, 13.53, 3.53]
nze_uced_nc = [11.72, 0.66, 17.16, 50.95, 6.16, 7.21, 10.25, 2.93]
nze_uced_dc = [11.11, 0.62, 16.78, 49.39, 8.53, 7.14, 9.89, 4.57]
nze_uced_sc = [11.15, 0.59, 16.25, 47.69, 10.57, 7.04, 10.34, 6.89]

# Porcentajes
porcentajes = {
    'UCED NC': ['↑28.2%', '↑73100%', '↑1.6%', '↓5.6%', '↑4.2%', '↓7.8%', '↓24.2%', '↓17.0%'],
    'UCED DC': ['↑21.6%', '↑68777%', '↓0.7%', '↓8.5%', '↑44.4%', '↓8.7%', '↓26.9%', '↑29.5%'],
    'UCED SC': ['↑22.0%', '↑65433%', '↓3.8%', '↓11.6%', '↑78.9%', '↓10.0%', '↓23.6%', '↑95.2%']
}

group1 = ['Hydro', 'Thermal', 'Wind', 'Solar']
group2 = ['Biomass', 'Geothermal', 'Batteries', 'Curtailment']
group1_idx = [technologies.index(t) for t in group1]
group2_idx = [technologies.index(t) for t in group2]

def extract_group(data, group_indices):
    return [data[i] for i in group_indices]

# Offset de texto para cada subplot
offsets = [8, 3]

fig, axes = plt.subplots(2, 1, figsize=(30, 10), sharex=False)
width = 0.23

def plot_group(ax, group, idxs, idx):
    x = np.arange(len(group))

    vals_esom = extract_group(nze_esom_nc, idxs)
    vals_uced_nc = extract_group(nze_uced_nc, idxs)
    vals_uced_dc = extract_group(nze_uced_dc, idxs)
    vals_uced_sc = extract_group(nze_uced_sc, idxs)

    bars1 = ax.bar(x - 1.5*width, vals_esom, width, label='NZE ESOM NC',
                   color=[colors[tech] for tech in group], alpha=0.4, edgecolor='black')
    bars2 = ax.bar(x - 0.5*width, vals_uced_nc, width, label='NZE UCED NC',
                   color=[colors[tech] for tech in group], alpha=0.6, edgecolor='black')
    bars3 = ax.bar(x + 0.5*width, vals_uced_dc, width, label='NZE UCED DC',
                   color=[colors[tech] for tech in group], alpha=0.8, edgecolor='black')
    bars4 = ax.bar(x + 1.5*width, vals_uced_sc, width, label='NZE UCED SC',
                   color=[colors[tech] for tech in group], alpha=1.0, edgecolor='black')

    ax.set_ylabel('Dispatch [TWh]', fontsize=20)
    ax.set_xticks(x)
    ax.set_xticklabels(group, fontsize=20)
    ax.tick_params(axis='y', labelsize=18)

    for i in range(len(group) - 1):
        ax.axvline(x=i + 0.5, color='gray', linestyle='--', linewidth=1)

    def add_labels(bars, values, label, porcentajes_local):
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                    f'{label}\n{val:.2f}', ha='center', va='bottom', fontsize=15, fontweight='bold')

            if label != 'ESOM' and porcentajes_local:
                perc = porcentajes_local[i]
                color = 'green' if '↑' in perc else 'red'
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offsets[idx],
                        perc, ha='center', va='bottom', fontsize=15, color=color)

    add_labels(bars1, vals_esom, "ESOM", None)
    add_labels(bars2, vals_uced_nc, "UCED NC", extract_group(porcentajes['UCED NC'], idxs))
    add_labels(bars3, vals_uced_dc, "UCED DC", extract_group(porcentajes['UCED DC'], idxs))
    add_labels(bars4, vals_uced_sc, "UCED SC", extract_group(porcentajes['UCED SC'], idxs))

    all_vals = vals_esom + vals_uced_nc + vals_uced_dc + vals_uced_sc
    ax.set_ylim(0, max(all_vals) + 8)

# Plot para cada grupo
plot_group(axes[0], group1, group1_idx, 0)
axes[0].set_title('ESCENARIO NZE: Dispatch Comparison by Technology (Group 1: Hydro - Solar)', fontsize=25)

plot_group(axes[1], group2, group2_idx, 1)
axes[1].set_title('ESCENARIO NZE: Dispatch Comparison by Technology (Group 2: Biomass - Curtailment)', fontsize=25)

plt.tight_layout()
plt.show()

#%% STACKED HORIZONTAL BARPLOT BY SCENARIO (EPI)
import matplotlib.pyplot as plt
import numpy as np

# Tecnologías y colores base
technologies = ['Hydro', 'Thermal', 'Wind', 'Solar', 'Biomass', 'Geothermal', 'Batteries', 'Curtailment']
base_colors = {
    'Hydro': '#00a0e1ff',
    'Thermal': '#d7642dff',
    'Wind': '#41afaaff',
    'Solar': '#e6a532ff',
    'Biomass': '#7daf4bff',
    'Geothermal': '#8B4513',
    'Batteries': '#57D53B',
    'Curtailment': 'red'
}

# Datos en TWh
data_dict = {
    'ESOM NC': [9.27, 0.045, 7.59, 19.73, 3.22, 8.17, 2.21, 5.87],
    'UCED NC': [11.46, 1.98, 8.37, 19.09, 1.22, 6.84, 1.64, 0.78],
    'UCED DC': [11.32, 2.22, 7.51, 17.63, 3.31, 6.68, 0.88, 2.88],
    'UCED SC': [10.91, 3.13, 6.18, 15.48, 6.73, 5.74, 0.57, 6.34]
}

# Diferencias porcentuales respecto a ESOM
porcentajes = {
    'UCED NC': ['↑23.6%', '↑4299%', '↑10.3%', '↓3.2%', '↓62.1%', '↓16.3%', '↓25.8%', '↓86.7%'],
    'UCED DC': ['↑22.1%', '↑4833%', '↓1.1%', '↓10.7%', '↑2.8%', '↓18.2%', '↓60.2%', '↓50.9%'],
    'UCED SC': ['↑17.7%', '↑6844%', '↓18.6%', '↓21.5%', '↑109%', '↓29.7%', '↓74.2%', '↑8.0%']
}

# Calcular porcentaje
totals = {k: sum(v) for k, v in data_dict.items()}
data_percent = {k: [100 * val / totals[k] for val in v] for k, v in data_dict.items()}

# Gráfico horizontal
scenarios = list(data_dict.keys())
y = np.arange(len(scenarios))
fig, ax = plt.subplots(figsize=(14, 8))
bar_height = 0.6
left = np.zeros(len(scenarios))

# Dibujar barras
for i, tech in enumerate(technologies):
    perc_vals = [data_percent[sc][i] for sc in scenarios]
    twh_vals = [data_dict[sc][i] for sc in scenarios]
    color = base_colors[tech]

    bars = ax.barh(y, perc_vals, height=bar_height, left=left,
                   label=tech, color=color, edgecolor='black', alpha=1)

    for j, bar in enumerate(bars):
        x_pos = bar.get_x() + bar.get_width()
        y_center = bar.get_y() + bar.get_height() / 2
        percent = perc_vals[j]
        twh = twh_vals[j]
        scenario = scenarios[j]

        # label_main = f"{percent:.1f}% ({twh:.2f} TWh)"

        # Posicionar texto
        # ax.text(x_pos + 1.0, y_center + 0.15, label_main, ha='left', va='center', fontsize=10)

        # if scenario != 'ESOM NC':
        #     diff = porcentajes[scenario][i]
        #     color_diff = 'green' if '↑' in diff else 'red'
        #     ax.text(x_pos + 1.0, y_center - 0.15, diff, ha='left', va='center',
        #             fontsize=10, color=color_diff, fontweight='bold')

    left += perc_vals

# Personalización del gráfico
ax.set_xlabel('Generation Share per technology [%]', fontsize=20)
ax.set_title('SCENARIO EPI: Dispatch Comparison by scenario and approach', fontsize=20)
ax.set_yticks(y)
ax.set_yticklabels(scenarios, fontsize=15)
ax.tick_params(axis='x', labelsize=15)
ax.set_xlim(0, 105)
ax.invert_yaxis()
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=15)
plt.tight_layout()
plt.show()


#%% STACKED HORIZONTAL BARPLOT BY SCENARIO (NZE)
import matplotlib.pyplot as plt
import numpy as np

# Tecnologías y colores base
technologies = ['Hydro', 'Thermal', 'Wind', 'Solar', 'Biomass', 'Geothermal', 'Batteries', 'Curtailment']
base_colors = {
    'Hydro': '#00a0e1ff',
    'Thermal': '#d7642dff',
    'Wind': '#41afaaff',
    'Solar': '#e6a532ff',
    'Biomass': '#7daf4bff',
    'Geothermal': '#8B4513',
    'Batteries': '#57D53B',
    'Curtailment': 'red'
}

# Datos en TWh
data_dict = {
    'ESOM NC': [9.14, 0.0009, 16.89, 53.97, 5.91, 7.82, 13.53, 3.53],
    'UCED NC': [11.72, 0.66, 17.16, 50.95, 6.16, 7.21, 10.25, 2.93],
    'UCED DC': [11.11, 0.62, 16.78, 49.39, 8.53, 7.14, 9.89, 4.57],
    'UCED SC': [11.15, 0.59, 16.25, 47.69, 10.57, 7.04, 10.34, 6.89]
}

# Diferencias porcentuales respecto a ESOM
porcentajes = {
    'UCED NC': ['↑28.2%', '↑73100%', '↑1.6%', '↓5.6%', '↑4.2%', '↓7.8%', '↓24.2%', '↓17.0%'],
    'UCED DC': ['↑21.6%', '↑68777%', '↓0.7%', '↓8.5%', '↑44.4%', '↓8.7%', '↓26.9%', '↑29.5%'],
    'UCED SC': ['↑22.0%', '↑65433%', '↓3.8%', '↓11.6%', '↑78.9%', '↓10.0%', '↓23.6%', '↑95.2%']
}

# Calcular porcentaje
totals = {k: sum(v) for k, v in data_dict.items()}
data_percent = {k: [100 * val / totals[k] for val in v] for k, v in data_dict.items()}

# Gráfico horizontal
scenarios = list(data_dict.keys())
y = np.arange(len(scenarios))
fig, ax = plt.subplots(figsize=(14, 8))
bar_height = 0.6
left = np.zeros(len(scenarios))

# Dibujar barras
for i, tech in enumerate(technologies):
    perc_vals = [data_percent[sc][i] for sc in scenarios]
    twh_vals = [data_dict[sc][i] for sc in scenarios]
    color = base_colors[tech]

    bars = ax.barh(y, perc_vals, height=bar_height, left=left,
                   label=tech, color=color, edgecolor='black', alpha=1)

    for j, bar in enumerate(bars):
        x_pos = bar.get_x() + bar.get_width()
        y_center = bar.get_y() + bar.get_height() / 2
        percent = perc_vals[j]
        twh = twh_vals[j]
        scenario = scenarios[j]

        # label_main = f"{percent:.1f}% ({twh:.2f} TWh)"

        # Posicionar texto
        # ax.text(x_pos + 1.0, y_center + 0.15, label_main, ha='left', va='center', fontsize=10)

        # if scenario != 'ESOM NC':
        #     diff = porcentajes[scenario][i]
        #     color_diff = 'green' if '↑' in diff else 'red'
        #     ax.text(x_pos + 1.0, y_center - 0.15, diff, ha='left', va='center',
        #             fontsize=10, color=color_diff, fontweight='bold')

    left += perc_vals

# Personalización del gráfico
ax.set_xlabel('Generation Share per technology [%]', fontsize=20)
ax.set_title('SCENARIO NZE: Dispatch Comparison by scenario and approach', fontsize=20)
ax.set_yticks(y)
ax.set_yticklabels(scenarios, fontsize=15)
ax.tick_params(axis='x', labelsize=15)
ax.set_xlim(0, 105)
ax.invert_yaxis()
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=15)
plt.tight_layout()
plt.show()

#%% STACKED HORIZONTAL BARPLOT BY SCENARIO (EPI) with FFR and PFR
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

# Tecnologías y colores
technologies = ['Hydro', 'Thermal', 'Wind', 'Solar', 'Biomass', 'Geothermal',
                'Batteries', 'Curtailment', 'FFR', 'PFR']
base_colors = {
    'Hydro': '#00a0e1ff',
    'Thermal': '#d7642dff',
    'Wind': '#41afaaff',
    'Solar': '#e6a532ff',
    'Biomass': '#7daf4bff',
    'Geothermal': '#8B4513',
    'Batteries': '#57D53B',
    'Curtailment': 'red',
    'FFR': '#6A5ACD',   # Azul violeta (SlateBlue)
    'PFR': '#20B2AA'    # Verde azulado (LightSeaGreen)
}

# Datos en TWh (con FFR y PFR al final)
data_dict = {
    'ESOM NC': [9.27, 0.045, 7.59, 19.73, 3.22, 8.17, 2.21, 5.87, 0, 0],
    'UCED NC': [11.46, 1.98, 8.37, 19.09, 1.22, 6.84, 1.64, 0.78, 0, 0],
    'UCED DC': [11.32, 2.22, 7.51, 17.63, 3.31, 6.68, 0.88, 2.88, 2.82, 0.76],
    'UCED SC': [10.91, 3.13, 6.18, 15.48, 6.73, 5.74, 0.57, 6.34, 4.34, 0.87]
}

# Índices de tecnologías que se consideran en la demanda (excluyen Curtailment, FFR, PFR)
techs_excluded = ['Curtailment', 'FFR', 'PFR']
included_indices = [i for i, tech in enumerate(technologies) if tech not in techs_excluded]

# Calcular demandas y porcentajes
data_percent = {}
demand_scenarios = {}
for scenario, values in data_dict.items():
    demand = sum([values[i] for i in included_indices])
    demand_scenarios[scenario] = demand
    data_percent[scenario] = [100 * val / demand for val in values]

# Gráfico
scenarios = list(data_dict.keys())
y = np.arange(len(scenarios))
fig, ax = plt.subplots(figsize=(14, 8))
bar_height = 0.6
left = np.zeros(len(scenarios))

for i, tech in enumerate(technologies):
    perc_vals = [data_percent[sc][i] for sc in scenarios]
    twh_vals = [data_dict[sc][i] for sc in scenarios]
    color = base_colors[tech]
    
    bars = ax.barh(y, perc_vals, height=bar_height, left=left,
                   label=tech, color=color, edgecolor='black', alpha=0.7)

    # Opcional: etiquetas internas si se desea
    # for j, bar in enumerate(bars):
    #     if perc_vals[j] > 1:
    #         ax.text(
    #             bar.get_x() + bar.get_width() / 2,
    #             bar.get_y() + bar.get_height() / 2,
    #             f"{twh_vals[j]:.1f} TWh\n({perc_vals[j]:.0f}%)",
    #             ha='center', va='center', fontsize=10, color='black'
    #         )

    left += perc_vals

# Línea negra sólida en 100% para cada barra
for idx in range(len(scenarios)):
    y_bottom = idx - bar_height / 2
    y_top = idx + bar_height / 2
    ax.vlines(x=100, ymin=y_bottom, ymax=y_top, colors='black',
              linestyles='solid', linewidth=5)

# Leyenda
demand_patch = mpatches.Patch(color='black', label='Demand')
handles, labels = ax.get_legend_handles_labels()
handles.append(demand_patch)
ax.legend(handles=handles, bbox_to_anchor=(1.005, 1.02), loc='upper left', fontsize=20)

# Ejes y estética
ax.set_xlabel('Generation Share and Reserves [%]', fontsize=20)
ax.set_title('SCENARIO EPI: Dispatch Comparison by Scenario and Approach', fontsize=26)
ax.set_yticks(y)
ax.set_yticklabels(scenarios, fontsize=20)
ax.tick_params(axis='x', labelsize=20)
ax.set_xlim(0, 130)
ax.invert_yaxis()

plt.tight_layout()
plt.show()


#%% STACKED HORIZONTAL BARPLOT BY SCENARIO (NZE) with FFR and PFR
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

# Tecnologías y colores base
technologies = ['Hydro', 'Thermal', 'Wind', 'Solar', 'Biomass', 'Geothermal', 'Batteries', 'Curtailment', 'FFR', 'PFR']
base_colors = {
    'Hydro': '#00a0e1ff',
    'Thermal': '#d7642dff',
    'Wind': '#41afaaff',
    'Solar': '#e6a532ff',
    'Biomass': '#7daf4bff',
    'Geothermal': '#8B4513',
    'Batteries': '#57D53B',
    'Curtailment': 'red',
    'FFR': '#6A5ACD',   # Azul violeta (SlateBlue)
    'PFR': '#20B2AA'    # Verde azulado (LightSeaGreen)
}

# Datos en TWh (incluyen FFR y PFR al final)
data_dict = {
    'ESOM NC': [9.14, 0.0009, 16.89, 53.97, 5.91, 7.82, 13.53, 3.53, 0, 0],
    'UCED NC': [11.72, 0.66, 17.16, 50.95, 6.16, 7.21, 10.25, 2.93, 0, 0],
    'UCED DC': [11.11, 0.62, 16.78, 49.39, 8.53, 7.14, 9.89, 4.57, 9.05, 0.89],
    'UCED SC': [11.15, 0.59, 16.25, 47.69, 10.57, 7.04, 10.34, 6.89, 9.71, 1.02]
}

# Calcular demanda neta (sin Curtailment, FFR ni PFR) y porcentajes
data_percent = {}
demand_scenarios = {}
for scenario, values in data_dict.items():
    demand = sum(values[:-3])  # Excluir Curtailment, FFR y PFR
    demand_scenarios[scenario] = demand
    data_percent[scenario] = [100 * val / demand for val in values]

# Gráfico
scenarios = list(data_dict.keys())
y = np.arange(len(scenarios))
fig, ax = plt.subplots(figsize=(14, 8))
bar_height = 0.6
left = np.zeros(len(scenarios))

for i, tech in enumerate(technologies):
    perc_vals = [data_percent[sc][i] for sc in scenarios]
    twh_vals = [data_dict[sc][i] for sc in scenarios]
    color = base_colors[tech]

    bars = ax.barh(y, perc_vals, height=bar_height, left=left,
                   label=tech, color=color, edgecolor='black', alpha=0.7)

    left += perc_vals

# Línea negra sólida al 100%
for idx in range(len(scenarios)):
    y_bottom = idx - bar_height / 2
    y_top = idx + bar_height / 2
    ax.vlines(x=100, ymin=y_bottom, ymax=y_top, colors='black',
              linestyles='solid', linewidth=5)

# Leyenda
demand_patch = mpatches.Patch(color='black', label='Demand')
handles, labels = ax.get_legend_handles_labels()
handles.append(demand_patch)
ax.legend(handles=handles, bbox_to_anchor=(1.005, 1.02), loc='upper left', fontsize=20)

# Personalización
ax.set_xlabel('Generation Share and Reserves [%]', fontsize=20)
ax.set_title('SCENARIO NZE: Dispatch Comparison by Scenario and Approach', fontsize=26)
ax.set_yticks(y)
ax.set_yticklabels(scenarios, fontsize=20)
ax.tick_params(axis='x', labelsize=20)
ax.set_xlim(0, 130)
ax.invert_yaxis()
plt.tight_layout()
plt.show()


#%% EPI CON VALORES
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

# Tecnologías y colores
technologies = ['Hydro', 'Thermal', 'Wind', 'Solar', 'Biomass', 'Geothermal',
                'Batteries', 'Curtailment', 'FFR', 'PFR']
base_colors = {
    'Hydro': '#00a0e1ff',
    'Thermal': '#d7642dff',
    'Wind': '#41afaaff',
    'Solar': '#e6a532ff',
    'Biomass': '#7daf4bff',
    'Geothermal': '#8B4513',
    'Batteries': '#57D53B',
    'Curtailment': 'red',
    'FFR': '#6A5ACD',
    'PFR': '#20B2AA'
}

# Datos en TWh
data_dict = {
    'ESOM NC': [9.27, 0.045, 7.59, 19.73, 3.22, 8.17, 2.21, 5.87, 0, 0],
    'UCED NC': [11.46, 1.98, 8.37, 19.09, 1.22, 6.84, 1.64, 0.78, 0, 0],
    'UCED DC': [11.32, 2.22, 7.51, 17.63, 3.31, 6.68, 0.88, 2.88, 2.82, 0.76],
    'UCED SC': [10.91, 3.13, 6.18, 15.48, 6.73, 5.74, 0.57, 6.34, 4.34, 0.87]
}

techs_excluded = ['Curtailment', 'FFR', 'PFR']
included_indices = [i for i, tech in enumerate(technologies) if tech not in techs_excluded]

# Calcular demanda total y porcentajes
data_percent = {}
demand_scenarios = {}
for scenario, values in data_dict.items():
    demand = sum([values[i] for i in included_indices])
    demand_scenarios[scenario] = demand
    data_percent[scenario] = [100 * val / demand for val in values]

# Plot
scenarios = list(data_dict.keys())
y = np.arange(len(scenarios)) * 1.2  # Aumenta separación vertical
fig, ax = plt.subplots(figsize=(16, 8))
bar_height = 0.6
left = np.zeros(len(scenarios))

inside_threshold = 6  # mínimo % para colocar dentro

# Para controlar alternancia de etiquetas pequeñas por fila
label_toggle = [0] * len(scenarios)

for i, tech in enumerate(technologies):
    perc_vals = [data_percent[sc][i] for sc in scenarios]
    twh_vals = [data_dict[sc][i] for sc in scenarios]
    color = base_colors[tech]

    bars = ax.barh(y, perc_vals, height=bar_height, left=left,
                   label=tech, color=color, edgecolor='black', alpha=0.7)

    for j, bar in enumerate(bars):
        if perc_vals[j] > 1:
            bar_center_x = bar.get_x() + bar.get_width() / 2
            bar_y_center = bar.get_y() + bar.get_height() / 2

            label_text = f"{perc_vals[j]:.1f}%\n{twh_vals[j]:.1f}TWh"

            if perc_vals[j] >= inside_threshold:
                # Etiqueta dentro
                ax.text(bar_center_x, bar_y_center, label_text,
                        ha='center', va='center', fontsize=14, color='black')
            else:
                # Alternar: si toggle par -> abajo; impar -> arriba
                offset = 0.31
                if label_toggle[j] % 2 == 0:
                    label_y = bar.get_y() - offset
                    va = 'top'
                else:
                    label_y = bar.get_y() + bar_height + offset
                    va = 'bottom'
                ax.text(bar_center_x, label_y, label_text,
                        ha='center', va=va, fontsize=14)
                label_toggle[j] += 1

    left += perc_vals

# Línea negra gruesa en 100%
for idx in range(len(scenarios)):
    y_bottom = idx - bar_height / 2*4
    y_top = idx + bar_height / 2 *4 
    ax.vlines(x=100, ymin=y_bottom, ymax=y_top, colors='black',
              linestyles='solid', linewidth=2, alpha=1)

# Leyenda
demand_patch = mpatches.Patch(color='black', label='Demand')
handles, labels = ax.get_legend_handles_labels()
handles.append(demand_patch)
ax.legend(handles=handles, bbox_to_anchor=(1.005, 1.02), loc='upper left', fontsize=16)

# Ejes y estética
ax.set_xlabel('Generation Share and Reserves [%]', fontsize=18)
ax.set_title('SCENARIO EPI: Dispatch Comparison by Scenario and Approach', fontsize=24)
ax.set_yticks(y)
ax.set_yticklabels(scenarios, fontsize=16)
ax.tick_params(axis='x', labelsize=16)
ax.set_xlim(0, 130)
ax.invert_yaxis()

# Ajustar márgenes superior e inferior del eje y
margin = 0.65  # puedes ajustar este valor
y_min = min(y) - margin
y_max = max(y) + margin
ax.set_ylim(y_max, y_min)  # ¡ojo! porque el eje está invertido

plt.tight_layout()
plt.show()


#%% NZE CON VALORES
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

# Tecnologías y colores
technologies = ['Hydro', 'Thermal', 'Wind', 'Solar', 'Biomass', 'Geothermal',
                'Batteries', 'Curtailment', 'FFR', 'PFR']
base_colors = {
    'Hydro': '#00a0e1ff',
    'Thermal': '#d7642dff',
    'Wind': '#41afaaff',
    'Solar': '#e6a532ff',
    'Biomass': '#7daf4bff',
    'Geothermal': '#8B4513',
    'Batteries': '#57D53B',
    'Curtailment': 'red',
    'FFR': '#6A5ACD',
    'PFR': '#20B2AA'
}

# Datos en TWh
data_dict = {
    'ESOM NC': [9.14, 0.0009, 16.89, 53.97, 5.91, 7.82, 13.53, 3.53, 0, 0],
    'UCED NC': [11.72, 0.66, 17.16, 50.95, 6.16, 7.21, 10.25, 2.93, 0, 0],
    'UCED DC': [11.11, 0.62, 16.78, 49.39, 8.53, 7.14, 9.89, 4.57, 9.05, 0.89],
    'UCED SC': [11.15, 0.59, 16.25, 47.69, 10.57, 7.04, 10.34, 6.89, 9.71, 1.02]
}

techs_excluded = ['Curtailment', 'FFR', 'PFR']
included_indices = [i for i, tech in enumerate(technologies) if tech not in techs_excluded]

# Calcular demanda total y porcentajes
data_percent = {}
demand_scenarios = {}
for scenario, values in data_dict.items():
    demand = sum([values[i] for i in included_indices])
    demand_scenarios[scenario] = demand
    data_percent[scenario] = [100 * val / demand for val in values]

# Plot
scenarios = list(data_dict.keys())
y = np.arange(len(scenarios)) * 1.2  # Aumenta separación vertical
fig, ax = plt.subplots(figsize=(16, 8))
bar_height = 0.6
left = np.zeros(len(scenarios))

inside_threshold = 6  # mínimo % para colocar dentro

# Para controlar alternancia de etiquetas pequeñas por fila
label_toggle = [0] * len(scenarios)

for i, tech in enumerate(technologies):
    perc_vals = [data_percent[sc][i] for sc in scenarios]
    twh_vals = [data_dict[sc][i] for sc in scenarios]
    color = base_colors[tech]

    bars = ax.barh(y, perc_vals, height=bar_height, left=left,
                   label=tech, color=color, edgecolor='black', alpha=0.7)

    for j, bar in enumerate(bars):
        if perc_vals[j] > 1:
            bar_center_x = bar.get_x() + bar.get_width() / 2
            bar_y_center = bar.get_y() + bar.get_height() / 2

            label_text = f"{perc_vals[j]:.1f}%\n{twh_vals[j]:.1f}TWh"

            if perc_vals[j] >= inside_threshold:
                # Etiqueta dentro
                ax.text(bar_center_x, bar_y_center, label_text,
                        ha='center', va='center', fontsize=14, color='black')
            else:
                # Alternar: si toggle par -> abajo; impar -> arriba
                offset = 0.31
                if label_toggle[j] % 2 == 0:
                    label_y = bar.get_y() - offset
                    va = 'top'
                else:
                    label_y = bar.get_y() + bar_height + offset
                    va = 'bottom'
                ax.text(bar_center_x, label_y, label_text,
                        ha='center', va=va, fontsize=14)
                label_toggle[j] += 1

    left += perc_vals

# Línea negra gruesa en 100%
for idx in range(len(scenarios)):
    y_bottom = idx - bar_height / 2*4
    y_top = idx + bar_height / 2 *4 
    ax.vlines(x=100, ymin=y_bottom, ymax=y_top, colors='black',
              linestyles='solid', linewidth=2, alpha=1)

# Leyenda
demand_patch = mpatches.Patch(color='black', label='Demand')
handles, labels = ax.get_legend_handles_labels()
handles.append(demand_patch)
ax.legend(handles=handles, bbox_to_anchor=(1.005, 1.02), loc='upper left', fontsize=16)

# Ejes y estética
ax.set_xlabel('Generation Share and Reserves [%]', fontsize=18)
ax.set_title('SCENARIO NZE: Dispatch Comparison by Scenario and Approach', fontsize=24)
ax.set_yticks(y)
ax.set_yticklabels(scenarios, fontsize=16)
ax.tick_params(axis='x', labelsize=16)
ax.set_xlim(0, 130)
ax.invert_yaxis()

# Ajustar márgenes superior e inferior del eje y
margin = 0.65  # puedes ajustar este valor
y_min = min(y) - margin
y_max = max(y) + margin
ax.set_ylim(y_max, y_min)  # ¡ojo! porque el eje está invertido

plt.tight_layout()
plt.show()
