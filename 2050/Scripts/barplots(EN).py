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
epi_capacities = [2165.71, 1404.74, 3298, 10787.38, 2220.14, 935, 904.87]

# Datos de capacidad instalada en MW por tecnología para el escenario NZE
nzetotal = 74326.86  # Suma de todas las capacidades para el escenario NZE
nze_capacities = [2165.71, 123.6, 7047, 28661.38, 4707.14, 935, 7507.16]

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


#%%
import matplotlib.pyplot as plt
import numpy as np

# Data
technologies = ['Hydroelectrics', 'Thermoelectrics', 'Wind', 'Solar', 'Biomass', 'Geothermal', 'Batteries']
epitotal = 28050.39
epi_capacities = [2165.71, 1404.74, 3298, 10787.38, 2220.14, 935, 7239.02]

nzetotal = 74326.86
nze_capacities = [2165.71, 123.6, 7047, 28661.38, 4707.14, 935, 60131.02]

# Percentage calculations
nze_percentages = [(cap / nzetotal) * 100 for cap in nze_capacities]
epi_percentages = [(cap / epitotal) * 100 for cap in epi_capacities]

# Define colors
colors = {
    'Hydroelectrics': '#0099cc',  # Blue
    'Thermoelectrics': '#e07b00',  # Orange
    'Wind': '#66cc66',             # Green
    'Solar': '#ffcc00',            # Yellow
    'Biomass': '#ff9966',          # Light Orange
    'Geothermal': '#800080',       # Purple
    'Batteries': '#66b3ff'         # Light Blue
}

# Set up the chart
x = np.arange(len(technologies))  # Bar positions
width = 0.45  # Bar width to avoid overlap

fig, ax = plt.subplots(figsize=(15, 5))  # A4 size (8.27 x 11.69 inches)

# EPI bars (more transparent)
bars_epi = ax.bar(x - width/2, epi_capacities, width, label='EPI', 
                    color=[colors[tech] for tech in technologies], alpha=0.6, edgecolor='black')

# NZE bars (more opaque)
bars_nze = ax.bar(x + width/2, nze_capacities, width, label='NZE', 
                    color=[colors[tech] for tech in technologies], alpha=1.0, edgecolor='black')

# Axis labels
ax.set_xlabel('Generation Technologies', fontsize=12)
ax.set_ylabel('Installed Capacity (MW)', fontsize=12)
ax.set_title('Installed Capacity Comparison by Technology (EPI vs NZE)', fontsize=14)

# X-axis labels
ax.set_xticks(x)
ax.set_xticklabels(technologies, fontsize=10)

# Separation lines between technologies
for i in range(len(technologies) - 1):
    ax.axvline(x=i + 0.5, color='gray', linestyle='--', linewidth=1)

# Function to add labels above the bars
def add_labels(bars, values, percentages, scenario):
    for bar, value, percentage in zip(bars, values, percentages):
        height = bar.get_height()
        
        # Constructing the text where only the scenario is in bold
        scenario_text = f'{scenario}'
        value_text = f'{value:.2f} MW'
        percentage_text = f'({percentage:.1f}%)'
        
        # Place the scenario text in bold and the rest in normal
        ax.text(bar.get_x() + bar.get_width()/2, height + 8000, 
                scenario_text, ha='center', va='bottom', fontsize=9, 
                fontweight='bold', color='black')  # Scenario in bold
        
        # Place the value and percentage text in normal font weight
        ax.text(bar.get_x() + bar.get_width()/2, height + 350, 
                f'{value_text}\n{percentage_text}', ha='center', va='bottom', 
                fontsize=9, fontweight='normal', color='black')

# Add labels above each bar
add_labels(bars_epi, epi_capacities, epi_percentages, "EPI")
add_labels(bars_nze, nze_capacities, nze_percentages, "NZE")


# Adjust the Y-axis scale
ax.set_yticks(np.arange(0, 90000, 5000))
ax.tick_params(axis='both', labelsize=10)

# Tight layout for better fit
plt.tight_layout()

# Show the plot
plt.show()

#%%
import matplotlib.pyplot as plt
import numpy as np

# Data
technologies = ['Hydroelectrics', 'Thermoelectrics', 'Wind', 'Solar', 'Biomass', 'Geothermal', 'Batteries']
epitotal = 21715.84
epi_capacities = [2165.71, 1404.74, 3298, 10787.38, 2220.14, 935, 904.87]

# Datos de capacidad instalada en MW por tecnología para el escenario NZE
nzetotal = 51146.99  # Suma de todas las capacidades para el escenario NZE
nze_capacities = [2165.71, 123.6, 7047, 28661.38, 4707.14, 935, 7507.16]

# Percentage calculations
nze_percentages = [(cap / nzetotal) * 100 for cap in nze_capacities]
epi_percentages = [(cap / epitotal) * 100 for cap in epi_capacities]

# Calculate percentage change
percentage_change = [(n - e) / e * 100 if e != 0 else 0 for e, n in zip(epi_capacities, nze_capacities)]

# Define colors
colors = {
    'Hydroelectrics': '#0099cc',  # Blue
    'Thermoelectrics': '#e07b00',  # Orange
    'Wind': '#66cc66',             # Green
    'Solar': '#ffcc00',            # Yellow
    'Biomass': '#ff9966',          # Light Orange
    'Geothermal': '#8B4513',       # Brown
    'Batteries': '#4B4B4B'        # Dark Gray
    # 'PumpedHydro': '#800080'      # Purple
}

# Set up the chart
x = np.arange(len(technologies))  # Bar positions
width = 0.45  # Bar width to avoid overlap

fig, ax = plt.subplots(figsize=(15, 5))  # A4 size (8.27 x 11.69 inches)

# EPI bars (more transparent)
bars_epi = ax.bar(x - width/2, epi_capacities, width, label='EPI', 
                    color=[colors[tech] for tech in technologies], alpha=0.6, edgecolor='black')

# NZE bars (more opaque)
bars_nze = ax.bar(x + width/2, nze_capacities, width, label='NZE', 
                    color=[colors[tech] for tech in technologies], alpha=1.0, edgecolor='black')

# Axis labels
ax.set_xlabel('Generation Technologies', fontsize=12)
ax.set_ylabel('Installed Capacity (MW)', fontsize=12)
ax.set_title('Installed Capacity Comparison by Technology (EPI vs NZE)', fontsize=14)

# X-axis labels
ax.set_xticks(x)
ax.set_xticklabels(technologies, fontsize=10)

# Separation lines between technologies
for i in range(len(technologies) - 1):
    ax.axvline(x=i + 0.5, color='gray', linestyle='--', linewidth=1)

# Function to add labels above the bars
def add_labels(bars, values, percentages, scenario):
    for bar, value, percentage in zip(bars, values, percentages):
        height = bar.get_height()
        
        # Constructing the text where only the scenario is in bold
        scenario_text = f'{scenario}'
        value_text = f'{value:.2f} MW'
        percentage_text = f'({percentage:.1f}%)'
        
        # Place the scenario text in bold and the rest in normal
        ax.text(bar.get_x() + bar.get_width()/2, height + 3000, 
                scenario_text, ha='center', va='bottom', fontsize=9, 
                fontweight='bold', color='black')  # Scenario in bold
        
        # Place the value and percentage text in normal font weight
        ax.text(bar.get_x() + bar.get_width()/2, height + 100, 
                f'{value_text}\n{percentage_text}', ha='center', va='bottom', 
                fontsize=9, fontweight='normal', color='black')

# Add labels above each bar
add_labels(bars_epi, epi_capacities, epi_percentages, "EPI")
add_labels(bars_nze, nze_capacities, nze_percentages, "NZE")

# Function to add percentage change and arrows
def add_change_labels(x_positions, epi_values, nze_values, percentage_changes):
    for x_pos, epi, nze, change in zip(x_positions, epi_values, nze_values, percentage_changes):
        if change > 0:
            arrow = '↑'  # Up arrow
            color = 'green'
        elif change < 0:
            arrow = '↓'  # Down arrow
            color = 'red'
        else:
            arrow = '='  # Right arrow (no change)
            color = 'gray'
        
        change_text = f'{change:.1f}% {arrow}'
        
        # Position the change label slightly below the NZE/EPI labels
        ax.text(x_pos, max(epi, nze) + 5000, change_text, 
                ha='center', va='bottom', fontsize=10, fontweight='bold', color=color)

# Add percentage change labels below the "EPI" and "NZE" labels
add_change_labels(x, epi_capacities, nze_capacities, percentage_change)

# Adjust the Y-axis scale
ax.set_yticks(np.arange(0, 45000, 5000))
ax.tick_params(axis='both', labelsize=10)

# Adjust X-axis limits to remove extra spacing on the sides
ax.set_xlim(-0.5, len(technologies) - 0.5)

# Tight layout for better fit
plt.tight_layout()

# Show the plot
plt.show()


#%%
import matplotlib.pyplot as plt
import numpy as np

# Data
technologies = ['Hydroelectrics', 'Thermoelectrics', 'Wind', 'Solar', 'Biomass', 'Geothermal', 'Batteries']
epitotal = 21715.84
epi_capacities = [2165.71, 1404.74, 3298, 10787.38, 2220.14, 935, 904.87]

# Datos de capacidad instalada en MW por tecnología para el escenario NZE
nzetotal = 51146.99  # Suma de todas las capacidades para el escenario NZE
nze_capacities = [2165.71, 123.6, 7047, 28661.38, 4707.14, 935, 7507.16]

# Percentage calculations
nze_percentages = [(cap / nzetotal) * 100 for cap in nze_capacities]
epi_percentages = [(cap / epitotal) * 100 for cap in epi_capacities]

# Calculate percentage change
percentage_change = [(n - e) / e * 100 if e != 0 else 0 for e, n in zip(epi_capacities, nze_capacities)]

# Define colors
colors = {
    'Hydroelectrics': '#00a0e1ff',  # Blue
    'Thermoelectrics': '#d7642dff',  # Orange
    'Wind': '#41afaaff',             # Green
    'Solar': '#e6a532ff',            # Yellow
    'Biomass': '#7daf4bff',          # Light Orange
    'Geothermal': '#8B4513',       # Brown
    'Batteries': '#57D53B'        # Dark Gray
    # 'PumpedHydro': '#800080'      # Purple
}

# Set up the chart
x = np.arange(len(technologies))  # Bar positions
width = 0.45  # Bar width to avoid overlap

fig, ax = plt.subplots(figsize=(15, 6))  # A4 size (8.27 x 11.69 inches)

# EPI bars (more transparent)
bars_epi = ax.bar(x - width/2, epi_capacities, width, label='EPI', 
                    color=[colors[tech] for tech in technologies], alpha=1, edgecolor='black')

# NZE bars (more opaque)
bars_nze = ax.bar(x + width/2, nze_capacities, width, label='NZE', 
                    color=[colors[tech] for tech in technologies], alpha=1, edgecolor='black')

# Axis labels
ax.set_xlabel('Generation Technologies', fontsize=12)
ax.set_ylabel('Installed Capacity (MW)', fontsize=12)
ax.set_title('Installed Capacity Comparison by Technology (EPI vs NZE)', fontsize=14)

# X-axis labels
ax.set_xticks(x)
ax.set_xticklabels(technologies, fontsize=10)

# Separation lines between technologies
for i in range(len(technologies) - 1):
    ax.axvline(x=i + 0.5, color='gray', linestyle='--', linewidth=1)

# Function to add labels above the bars
def add_labels(bars, values, percentages, scenario):
    for bar, value, percentage in zip(bars, values, percentages):
        height = bar.get_height()
        
        # Constructing the text where only the scenario is in bold
        scenario_text = f'{scenario}'
        value_text = f'{value:.2f} MW'
        percentage_text = f'({percentage:.1f}%)'
        
        # Place the scenario text in bold and the rest in normal
        ax.text(bar.get_x() + bar.get_width()/2, height + 3000, 
                scenario_text, ha='center', va='bottom', fontsize=9, 
                fontweight='bold', color='black')  # Scenario in bold
        
        # Place the value and percentage text in normal font weight
        ax.text(bar.get_x() + bar.get_width()/2, height + 100, 
                f'{value_text}\n{percentage_text}', ha='center', va='bottom', 
                fontsize=9, fontweight='normal', color='black')

# Add labels above each bar
add_labels(bars_epi, epi_capacities, epi_percentages, "EPI")
add_labels(bars_nze, nze_capacities, nze_percentages, "NZE")

# Function to add percentage change and arrows
def add_change_labels(x_positions, epi_values, nze_values, percentage_changes):
    for x_pos, epi, nze, change in zip(x_positions, epi_values, nze_values, percentage_changes):
        if change > 0:
            arrow = '↑'  # Up arrow
            color = 'green'
        elif change < 0:
            arrow = '↓'  # Down arrow
            color = 'red'
        else:
            arrow = '='  # Right arrow (no change)
            color = 'gray'
        
        change_text = f'{change:.1f}% {arrow}'
        
        # Position the change label slightly below the NZE/EPI labels
        ax.text(x_pos, max(epi, nze) + 5000, change_text, 
                ha='center', va='bottom', fontsize=10, fontweight='bold', color=color)

# Add percentage change labels below the "EPI" and "NZE" labels
add_change_labels(x, epi_capacities, nze_capacities, percentage_change)

# Adjust the Y-axis scale
ax.set_yticks(np.arange(0, 45000, 5000))
ax.tick_params(axis='both', labelsize=10)

# Adjust X-axis limits to remove extra spacing on the sides
ax.set_xlim(-0.5, len(technologies) - 0.5)

# Tight layout for better fit
plt.tight_layout()

# Show the plot
plt.show()
#%% PRESENTACION ECOS
import matplotlib.pyplot as plt
import numpy as np

# Data
technologies = ['Hydroelectrics', 'Thermoelectrics', 'Wind', 'Solar', 'Biomass', 'Geothermal', 'Batteries']
epitotal = 21716
epi_capacities = [2166, 1405, 3298, 10787, 2220, 935, 905]

# Datos de capacidad instalada en MW por tecnología para el escenario NZE
nzetotal = 51147  # Suma de todas las capacidades para el escenario NZE
nze_capacities = [2166, 124, 7047, 28661, 4707, 935, 7507]

# Percentage calculations
nze_percentages = [(cap / nzetotal) * 100 for cap in nze_capacities]
epi_percentages = [(cap / epitotal) * 100 for cap in epi_capacities]

# Calculate percentage change
percentage_change = [(n - e) / e * 100 if e != 0 else 0 for e, n in zip(epi_capacities, nze_capacities)]

# Define colors
colors = {
    'Hydroelectrics': '#00a0e1ff',  # Blue
    'Thermoelectrics': '#d7642dff',  # Orange
    'Wind': '#41afaaff',             # Green
    'Solar': '#e6a532ff',            # Yellow
    'Biomass': '#7daf4bff',          # Light Orange
    'Geothermal': '#8B4513',       # Brown
    'Batteries': '#57D53B'        # Dark Gray
    # 'PumpedHydro': '#800080'      # Purple
}

# Set up the chart
x = np.arange(len(technologies))  # Bar positions
width = 0.45  # Bar width to avoid overlap

fig, ax = plt.subplots(figsize=(18, 8))  # A4 size (8.27 x 11.69 inches)

# EPI bars (more transparent)
bars_epi = ax.bar(x - width/2, epi_capacities, width, label='EPI', 
                    color=[colors[tech] for tech in technologies], alpha=1, edgecolor='black')

# NZE bars (more opaque)
bars_nze = ax.bar(x + width/2, nze_capacities, width, label='NZE', 
                    color=[colors[tech] for tech in technologies], alpha=1, edgecolor='black')

# Axis labels
ax.set_xlabel('Generation Technologies', fontsize=20)
ax.set_ylabel('Installed Capacity (MW)', fontsize=20)
ax.set_title('Installed Capacity Comparison by Technology (EPI vs NZE)', fontsize=24)

# X-axis labels
ax.set_xticks(x)
ax.set_xticklabels(technologies, fontsize=18)

# Separation lines between technologies
for i in range(len(technologies) - 1):
    ax.axvline(x=i + 0.5, color='gray', linestyle='--', linewidth=1)

# Function to add labels above the bars
def add_labels(bars, values, percentages, scenario):
    for bar, value, percentage in zip(bars, values, percentages):
        height = bar.get_height()
        
        # Constructing the text where only the scenario is in bold
        scenario_text = f'{scenario}'
        value_text = f'{value:.0f}MW'
        percentage_text = f'({percentage:.0f}%)'
        
        # Place the scenario text in bold and the rest in normal
        ax.text(bar.get_x() + bar.get_width()/2, height + 3500, 
                scenario_text, ha='center', va='bottom', fontsize=18, 
                fontweight='bold', color='black')  # Scenario in bold
        
        # Place the value and percentage text in normal font weight
        ax.text(bar.get_x() + bar.get_width()/2, height + 100, 
                f'{value_text}\n{percentage_text}', ha='center', va='bottom', 
                fontsize=18, fontweight='normal', color='black')

# Add labels above each bar
add_labels(bars_epi, epi_capacities, epi_percentages, "EPI")
add_labels(bars_nze, nze_capacities, nze_percentages, "NZE")

# Function to add percentage change and arrows
def add_change_labels(x_positions, epi_values, nze_values, percentage_changes):
    for x_pos, epi, nze, change in zip(x_positions, epi_values, nze_values, percentage_changes):
        if change > 0:
            arrow = '↑'  # Up arrow
            color = 'green'
        elif change < 0:
            arrow = '↓'  # Down arrow
            color = 'red'
        else:
            arrow = '='  # Right arrow (no change)
            color = 'gray'
        
        change_text = f'{change:.1f}% {arrow}'
        
        # Position the change label slightly below the NZE/EPI labels
        ax.text(x_pos, max(epi, nze) + 5000, change_text, 
                ha='center', va='bottom', fontsize=18, color=color)

# Add percentage change labels below the "EPI" and "NZE" labels
add_change_labels(x, epi_capacities, nze_capacities, percentage_change)

# Adjust the Y-axis scale
ax.set_yticks(np.arange(0, 45000, 5000))
ax.tick_params(axis='both', labelsize=18)

# Adjust X-axis limits to remove extra spacing on the sides
ax.set_xlim(-0.5, len(technologies) - 0.5)

# Tight layout for better fit
plt.tight_layout()

# Show the plot
plt.show()


#%%
import matplotlib.pyplot as plt
import numpy as np

# Datos de demanda para cada sector
sectors = ['Residential Cooking', 'Residential Waterheating', 'Residential Refrigeration', 'Residential Lighting',
           'Residential Spacecooling', 'Residential Spaceheating', 'Residential Otherelectronics',
           'Commercial Electricity', 'Commercial Cooking', 'Commercial Waterheating', 'Commercial Mechanicalenergy',
           'Commercial Lighting', 'Industrial Processheat', 'Industrial Mechanicalenergy', 'Industrial Lighting',
           'Transport Passenger', 'Transport Freight', 'Agriculture Mechanicalenergy', 'Agriculture Drivingforce',
           'Mining Mechanicalenergy', 'Fishing&Others Mechanicalenergy', 'PublicLighting', 'Non-energy Processheat']

# Energía de cada sector (demanda en TWh)
sector_demands = {
    'Residential Cooking': 4.74,
    'Residential Waterheating': 1.17,
    'Residential Refrigeration': 0.34,
    'Residential Lighting': 48.13,
    'Residential Spacecooling': 0.10,
    'Residential Spaceheating': 21.25,
    'Residential Otherelectronics': 0.82,
    'Commercial Electricity': 1.16,
    'Commercial Cooking': 0.30,
    'Commercial Waterheating': 0.14,
    'Commercial Mechanicalenergy': 77.00,
    'Commercial Lighting': 30.14,
    'Industrial Processheat': 13.19,
    'Industrial Mechanicalenergy': 1.58,
    'Industrial Lighting': 26.43,
    'Transport Passenger': 0.35,
    'Transport Freight': 15.24,
    'Agriculture Mechanicalenergy': 0.23,
    'Agriculture Drivingforce': 50.12,
    'Mining Mechanicalenergy': 1.55,
    'Fishing&Others Mechanicalenergy': 0.45,
    'PublicLighting': 54.13,
    'Non-energy Processheat': 1.76,
}

# Zonas de suministro: CE, OR, NO, SU
zones = ['CE', 'OR', 'NO', 'SU']

# Generar distribución de energía para NZE y EPI
def generate_energy_distribution(demand, scenario):
    total_demand = demand
    if scenario == 'NZE':
        # Escenario NZE: 40% energía primaria, 60% distribuido entre las zonas
        primary_energy = 0.4 * total_demand
        zone_energy = 0.6 * total_demand
    elif scenario == 'EPI':
        # Escenario EPI: 30% energía primaria, 70% distribuido entre las zonas
        primary_energy = 0.3 * total_demand
        zone_energy = 0.7 * total_demand

    # Distribuir energía entre las zonas de forma aleatoria
    zone_distribution = np.random.rand(4)  # 4 zonas (CE, OR, NO, SU)
    zone_distribution = zone_distribution / zone_distribution.sum()  # Normalizamos para que sume 1
    zone_energy_distribution = zone_distribution * zone_energy  # Asignamos la energía entre las zonas
    
    return primary_energy, zone_energy_distribution

# Visualización
fig, ax = plt.subplots(2, 1, figsize=(10, 12))

# Crear gráfico para los escenarios
for i, scenario in enumerate(['NZE', 'EPI']):
    bottom = np.zeros(len(sectors))
    ax[i].set_title(f'Energy Demand Allocation by Sector ({scenario} Scenario)')
    ax[i].set_ylabel('Energy (TWh)')
    
    for j, sector in enumerate(sectors):
        demand = sector_demands[sector]
        primary_energy, zone_energy_distribution = generate_energy_distribution(demand, scenario)
        
        # Primero, dibujamos la energía primaria
        ax[i].bar(sector, primary_energy, bottom=bottom[j], color='gray', label=f'Primary Energy ({scenario})' if j == 0 else "")
        bottom[j] += primary_energy
        
        # Luego, dibujamos la energía de las zonas
        for k, zone in enumerate(zones):
            ax[i].bar(sector, zone_energy_distribution[k], bottom=bottom[j], label=f'Zone {zone} ({scenario})' if j == 0 else "", color=plt.cm.Paired(k / len(zones)))
            bottom[j] += zone_energy_distribution[k]

    ax[i].legend(title='Energy Supply')
    ax[i].tick_params(axis='x', rotation=90)

plt.tight_layout()
plt.show()

#%%
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

# Datos de demanda para cada sector en TWh
sector_demands = {
    'Residential Cooking': 4.74,
    'Residential Waterheating': 1.17,
    'Residential Refrigeration': 0.34,
    'Residential Lighting': 48.13,
    'Residential Spacecooling': 0.10,
    'Residential Spaceheating': 21.25,
    'Residential Otherelectronics': 0.82,
    'Commercial Electricity': 1.16,
    'Commercial Cooking': 0.30,
    'Commercial Waterheating': 0.14,
    'Commercial Mechanicalenergy': 77.00,
    'Commercial Lighting': 30.14,
    'Industrial Processheat': 13.19,
    'Industrial Mechanicalenergy': 1.58,
    'Industrial Lighting': 26.43,
    'Transport Passenger': 0.35,
    'Transport Freight': 15.24,
    'Agriculture Mechanicalenergy': 0.23,
    'Agriculture Drivingforce': 50.12,
    'Mining Mechanicalenergy': 1.55,
    'Fishing&Others Mechanicalenergy': 0.45,
    'PublicLighting': 54.13,
    'Non-energy Processheat': 1.76,
}

# Clasificación en cinco rangos
very_high_demand = {k: v for k, v in sector_demands.items() if v >= 50}
high_demand = {k: v for k, v in sector_demands.items() if 25 <= v < 50}
medium_demand = {k: v for k, v in sector_demands.items() if 10 <= v < 25}
low_demand = {k: v for k, v in sector_demands.items() if 2 <= v < 10}
very_low_demand = {k: v for k, v in sector_demands.items() if v < 2}

# Colores
zone_color = '#A6A6A6'  # Color para las zonas (gris claro)
primary_color = '#4B4B4B'  # Color para la energía primaria (gris oscuro)

# Nombres de las zonas
zone_labels = ["CE", "OR", "NO", "SU"]

# Función para generar distribución de energía
def generate_energy_distribution(demand, scenario):
    if scenario == 'NZE':
        primary_energy = 0.4 * demand
        zone_energy = 0.6 * demand
    else:  # 'EPI'
        primary_energy = 0.3 * demand
        zone_energy = 0.7 * demand

    zone_distribution = np.array([0.25, 0.25, 0.25, 0.25]) * zone_energy
    return primary_energy, zone_distribution

# Función para graficar sectores
def plot_sectors(sector_dict, title):
    if not sector_dict:
        return  # No grafica si el diccionario está vacío

    fig, ax = plt.subplots(figsize=(12, len(sector_dict) * 0.5))
    y_pos = np.arange(len(sector_dict))

    for i, (sector, demand) in enumerate(sector_dict.items()):
        primary_energy_NZE, zone_energy_NZE = generate_energy_distribution(demand, 'NZE')
        primary_energy_EPI, zone_energy_EPI = generate_energy_distribution(demand, 'EPI')

        # Barras para NZE
        ax.barh(y_pos[i] - 0.2, primary_energy_NZE, height=0.4, color=primary_color, edgecolor='black', linewidth=1)
        bottom_NZE = primary_energy_NZE
        for k in range(4):
            ax.barh(y_pos[i] - 0.2, zone_energy_NZE[k], height=0.4, left=bottom_NZE, color=zone_color, edgecolor='black', linewidth=0.8)
            ax.text(bottom_NZE + zone_energy_NZE[k] / 2, y_pos[i] - 0.2, zone_labels[k], ha='center', va='center', fontsize=8, color='black')
            bottom_NZE += zone_energy_NZE[k]
        ax.text(bottom_NZE + 0.1, y_pos[i] - 0.2, 'NZE', va='center', fontsize=9, color='black')

        # Barras para EPI
        ax.barh(y_pos[i] + 0.2, primary_energy_EPI, height=0.4, color=primary_color, edgecolor='black', linewidth=1)
        bottom_EPI = primary_energy_EPI
        for k in range(4):
            ax.barh(y_pos[i] + 0.2, zone_energy_EPI[k], height=0.4, left=bottom_EPI, color=zone_color, edgecolor='black', linewidth=0.8)
            ax.text(bottom_EPI + zone_energy_EPI[k] / 2, y_pos[i] + 0.2, zone_labels[k], ha='center', va='center', fontsize=8, color='black')
            bottom_EPI += zone_energy_EPI[k]
        ax.text(bottom_EPI + 0.1, y_pos[i] + 0.2, 'EPI', va='center', fontsize=9, color='black')

    # Títulos y etiquetas
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Energía Total (TWh)', fontsize=12)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sector_dict.keys(), fontsize=9)

    # Leyenda
    handles = [
        mpatches.Patch(color=primary_color, edgecolor='black', linewidth=1, label='Primary Energy'),
        mpatches.Patch(color=zone_color, edgecolor='black', linewidth=0.8, label='Zonas (divididas por líneas)')
    ]
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1, 1), title="Tipos de Energía")

    plt.tight_layout()
    plt.show()

# Graficar diferentes categorías de demanda
plot_sectors(very_high_demand, 'Sectores con Muy Alta Demanda de Energía')
plot_sectors(high_demand, 'Sectores con Alta Demanda de Energía')
plot_sectors(medium_demand, 'Sectores con Demanda Media de Energía')
plot_sectors(low_demand, 'Sectores con Baja Demanda de Energía')
plot_sectors(very_low_demand, 'Sectores con Muy Baja Demanda de Energía')
#%%
import matplotlib.pyplot as plt
import numpy as np

# Data
years = ["2023", "2030", "2050"]
capacities = [120, 194, 400]  # Power in MW
names = ["Misicuni", "Sehuencas", "Rositas"]

# Increase in percentage compared to Misicuni
percentages = ["+0.0%", "+61.7%", "+233.3%"]

# Bar colors
colors = ["#7589A2", "#325C66", "#1E3F3F"]

# Create figure and axes
fig, ax = plt.subplots(figsize=(5, 2))

# Bar width
bar_width = 0.6

# Create bars
bars = ax.bar(years, capacities, width=bar_width, color=colors, edgecolor="black")

# Labels above the bars
for bar, name, cap, perc in zip(bars, names, capacities, percentages):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 10,
            f"{cap} MW\n{perc}",
            ha='center', fontsize=9, fontweight="normal")
    
    ax.text(bar.get_x() + bar.get_width()/2, height / 2,
            name, ha='center', fontsize=10, fontweight="bold", color="black")

# Axis labels
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Power (MW)", fontsize=12)
ax.set_title("Largest installed generator each year", fontsize=13)

# X-axis customization
ax.set_xticks(range(len(years)))
ax.set_xticklabels(years, fontsize=10)

# Y-axis grid lines
ax.grid(axis="y", linestyle="--", alpha=0.6)

# Set Y-axis limit
ax.set_ylim(0, 600)

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()
#%%
import matplotlib.pyplot as plt

# Data
capacity = [800, 1200, 1800, 2500, 3200, 4000, 5000]  # MW
penetration_no_constraints = [15, 25, 38, 47, 53, 58, 60]  # % 
penetration_static_constraints = [10, 18, 26, 30, 34, 37, 39]  # % 

# Create figure
plt.figure(figsize=(7, 5))

# Plot both curves
plt.plot(capacity, penetration_no_constraints, 'bo-', label="No Stability Constraints")  # Blue with circles 
plt.plot(capacity, penetration_static_constraints, 'g^-', label="Static Constraints")  # Green with triangles 

# Labels and title
plt.xlabel("Variable Renewable Energy Capacity (MW)", fontsize=12)
plt.ylabel("Renewable Energy Penetration (%)", fontsize=12)
plt.title("Saturation Curve of Variable Renewable Energy Penetration", fontsize=13)

# Grid and legend
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=10, loc="upper left")

# Show plot
plt.show()
#%%
import matplotlib.pyplot as plt
import numpy as np

# Data
capacity = [800, 1200, 1800, 2500, 3200, 4000, 5000]  # MW
penetration_no_constraints = [15, 25, 38, 47, 53, 58, 60]  # % 
penetration_static_constraints = [10, 18, 26, 30, 34, 37, 39]  # % 

# Generate dynamic constraints data (closer to no constraints but randomized)
np.random.seed(42)  # For reproducibility
penetration_dynamic_constraints = [
    penetration_no_constraints[i] - np.random.randint(2, 6) for i in range(len(capacity))
]

# Create figure
plt.figure(figsize=(7, 5))

# Plot all curves
plt.plot(capacity, penetration_no_constraints, 'bo-', label="No Stability Constraints")  # Blue with circles 
plt.plot(capacity, penetration_static_constraints, 'g^-', label="Static Constraints")  # Green with triangles 
plt.plot(capacity, penetration_dynamic_constraints, 'rs-', label="Dynamic Constraints")  # Red with squares

# Labels and title
plt.xlabel("Variable Renewable Energy Capacity (MW)", fontsize=12)
plt.ylabel("Renewable Energy Penetration (%)", fontsize=12)
plt.title("Saturation Curve of Variable Renewable Energy Penetration", fontsize=13)

# Grid and legend
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=10, loc="upper left")

# Show plot
plt.show()

