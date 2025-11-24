# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 12:55:02 2025

@author: navia
"""

#%% DEMAND STACKED HORIZONTAL BARPLOT BY SCENARIO
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

# Tecnologías y colores base
sector = ['Residential', 'Commercial', 'Industrial', 'Transport', 'Agriculture', 'Mining', 'Fishing&Others', 'PublicLighting', 'Non-energy']
base_colors = {
    'Residential': '#4B9CD3',
    'Commercial': '#F4A261',
    'Industrial': '#6D6875',
    'Transport': '#E9C46A',
    'Agriculture': '#81B29A',
    'Mining': '#8D6E63',
    'Fishing&Others': '#3AAFA9',
    'PublicLighting': '#A8A8A8',
    'Non-energy': '#9A8C98'
}

# Datos en TWh (incluyen Non-energy y PFR al final)
data_dict = {
    'EPI': [9.14, 0.0009, 16.89, 53.97, 5.91, 7.82, 13.53, 3.53, 0],
    'NZE': [11.72, 0.66, 17.16, 50.95, 6.16, 7.21, 10.25, 2.93, 0],
}

# Calcular demanda neta (sin PublicLighting, Non-energy ni PFR) y porcentajes
data_percent = {}
demand_scenarios = {}
for scenario, values in data_dict.items():
    demand = sum(values[:-3])  # Excluir PublicLighting, Non-energy y PFR
    demand_scenarios[scenario] = demand
    data_percent[scenario] = [100 * val / demand for val in values]

# Gráfico
scenarios = list(data_dict.keys())
y = np.arange(len(scenarios))
fig, ax = plt.subplots(figsize=(14, 8))
bar_height = 0.6
left = np.zeros(len(scenarios))

for i, tech in enumerate(sector):
    perc_vals = [data_percent[sc][i] for sc in scenarios]
    twh_vals = [data_dict[sc][i] for sc in scenarios]
    color = base_colors[tech]

    bars = ax.barh(y, perc_vals, height=bar_height, left=left,
                   label=tech, color=color, edgecolor='black', alpha=0.7)

    left += perc_vals

# # Línea negra sólida al 100%
# for idx in range(len(scenarios)):
#     y_bottom = idx - bar_height / 2
#     y_top = idx + bar_height / 2
#     ax.vlines(x=100, ymin=y_bottom, ymax=y_top, colors='black',
#               linestyles='solid', linewidth=5)

# Leyenda
# demand_patch = mpatches.Patch(color='black', label='Demand')
handles, labels = ax.get_legend_handles_labels()
# handles.append(demand_patch)
ax.legend(handles=handles, bbox_to_anchor=(1.005, 1.02), loc='upper left', fontsize=20)

# Personalización
ax.set_xlabel('Demand per sector [%]', fontsize=20)
ax.set_title('SCENARIO NZE: Dispatch Comparison by Scenario and Approach', fontsize=26)
ax.set_yticks(y)
ax.set_yticklabels(scenarios, fontsize=20)
ax.tick_params(axis='x', labelsize=20)
ax.set_xlim(0, 100)
ax.invert_yaxis()
plt.tight_layout()
plt.show()
