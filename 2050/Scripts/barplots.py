# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 10:04:23 2024

@author: navia
"""
import matplotlib.pyplot as plt
import numpy as np

# Datos de capacidad instalada en MW por tecnología para el escenario EPI
technologies = ['Hidroeléctricas', 'Termoeléctricas', 'Eólicas', 'Solar', 'Biomasa', 'Geotérmica', 'Baterías']
epitotal = 28050.39
epi_capacities = [2165.71, 1404.74, 3298, 10787.38, 2220.14, 935, 7239.02]

# Datos de capacidad instalada en MW por tecnología para el escenario NZE
nzetotal = 74326.86  # Suma de todas las capacidades para el escenario NZE
nze_capacities = [2165.71, 123.6, 7047, 28661.38, 4707.14, 935, 60131.02]

# Calcular los porcentajes para el escenario NZE
nze_percentages = [(cap / nzetotal) * 100 for cap in nze_capacities]
epi_percentages = [(cap / epitotal) * 100 for cap in epi_capacities]

# Colores para las diferentes tecnologías (inspirados en los reportes de IEA)
colors = {
    'Hidroeléctricas': '#0099cc',  # Azul
    'Termoeléctricas': '#e07b00',  # Naranja
    'Eólicas': '#66cc66',          # Verde
    'Solar': '#ffcc00',            # Amarillo
    'Biomasa': '#ff9966',          # Naranja claro
    'Geotérmica': '#800080',       # Púrpura
    'Baterías': '#66b3ff'          # Azul claro
}

# Crear el gráfico de barras para el escenario EPI
x = np.arange(len(technologies))  # Ubicación de las barras
width = 0.35  # Ancho de las barras (ajustado para mayor separación)

fig, ax = plt.subplots(figsize=(14, 7))

# Barras para el escenario EPI (en MW)
bars_epi = ax.bar(x, epi_capacities, width, label='Escenario EPI (MW)', color=[colors[tech] for tech in technologies])

# Añadir etiquetas, título y leyenda para EPI
ax.set_xlabel('Tecnologías de Generación')
ax.set_ylabel('Capacidad (MW)')
ax.set_title('Capacidad Instalada por Tecnología en el Escenario EPI')

# Reajustar los xticks para evitar que se sobrepongan
ax.set_xticks(x)
ax.set_xticklabels(technologies)  # Usamos solo los nombres de tecnologías

# Añadir líneas horizontales en los yticks
ax.yaxis.grid(True, linestyle='--', linewidth=0.5)

# Función para agregar los valores y los porcentajes encima de las barras
def add_labels(bars, values, percentages, is_percentage=False):
    for bar, value, percentage in zip(bars, values, percentages):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 100, 
                f'{value:.2f} MW\n({percentage:.2f}%)', ha='center', va='bottom', fontsize=10)

# Agregar las etiquetas de los valores a las barras para EPI
add_labels(bars_epi, epi_capacities, epi_percentages)

# Ajustar los yticks para EPI (cada 1000 MW hasta 15000 MW)
yticks_values_epi = np.arange(0, 16000, 1000)  # YTICKS cada 1000 MW hasta 15000 MW
ax.set_yticks(yticks_values_epi)
ax.set_yticklabels([f'{int(val)} MW' for val in yticks_values_epi])

# Mostrar gráfico para EPI
plt.tight_layout()
plt.show()

# Crear el gráfico de barras para el escenario NZE
fig, ax = plt.subplots(figsize=(14, 7))

# Barras para el escenario NZE (en MW)
bars_nze = ax.bar(x, nze_capacities, width, label='Escenario NZE (MW)', color=[colors[tech] for tech in technologies])

# Añadir etiquetas, título y leyenda para NZE
ax.set_xlabel('Tecnologías de Generación')
ax.set_ylabel('Capacidad (MW)')
ax.set_title('Capacidad Instalada por Tecnología en el Escenario NZE')

# Reajustar los xticks para evitar que se sobrepongan
ax.set_xticks(x)
ax.set_xticklabels(technologies)  # Usamos solo los nombres de tecnologías

# Añadir líneas horizontales en los yticks
ax.yaxis.grid(True, linestyle='--', linewidth=0.5)

# Agregar las etiquetas de los valores a las barras para NZE
add_labels(bars_nze, nze_capacities, nze_percentages)

# Ajustar los yticks para NZE (cada 5000 MW hasta 65000 MW)
yticks_values_nze = np.arange(0, 70000, 5000)  # YTICKS cada 5000 MW hasta 65000 MW
ax.set_yticks(yticks_values_nze)
ax.set_yticklabels([f'{int(val)} MW' for val in yticks_values_nze])

# Mostrar gráfico para NZE
plt.tight_layout()
plt.show()
