# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 15:08:44 2023

@author: navia
"""
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.lines import Line2D

# Cargar el shapefile de Bolivia con los departamentos
bolivia_departamentos = gpd.read_file('C:/Users/navia/Documents/DISPASET MODELS/5.12ENERGY/Maps/Bolivia/municipios339.shp')

# Definir los colores serios para cada zona
colors = {
    'La Paz': '#4F5D75',   # Azul oscuro
    'Beni': '#4F5D75',     # Azul oscuro
    'Pando': '#4F5D75',    # Azul oscuro
    'Cochabamba': 'blue',  # Gris violáceo
    'Oruro': 'blue',       # Gris violáceo
    'Santa Cruz': '#BC6C25',  # Marrón tierra
    'Tarija': '#E5989B',      # Rosa pálido oscuro
    'Chuquisaca': '#E5989B',  # Rosa pálido oscuro
    'PotosÃ­': '#E5989B'       # Rosa pálido oscuro
}

# Lidiar con valores NaN asignándoles un color por defecto (gris claro)
bolivia_departamentos['color'] = bolivia_departamentos['departamen'].map(colors).fillna('#D3D3D3')

# Definir las zonas
zona_norte = ['La Paz', 'Beni', 'Pando']
zona_central = ['Cochabamba', 'Oruro']
zona_oriental = ['Santa Cruz']
zona_sur = ['PotosÃ­', 'Chuquisaca', 'Tarija']

# Crear una nueva columna para las zonas
bolivia_departamentos['zona'] = bolivia_departamentos['departamen'].apply(
    lambda x: 'Norte' if x in zona_norte else 
              'Central' if x in zona_central else 
              'Oriental' if x in zona_oriental else 
              'Sur' if x in zona_sur else None
)

# Cargar los datos de líneas de transmisión y plantas de energía
df_power_plants = pd.read_csv('custom_powerplants.csv')
gdf_transmission_lines = gpd.read_file('../Bolivia-DataBase-main/transmision_sin_20190416/transmision_sin_20190416.shp')

# Cargar el shapefile de países para filtrar Bolivia
world = gpd.read_file('C:/Users/navia/Documents/DISPASET MODELS/5.12ENERGY/Maps/Countries/ne_10m_admin_0_countries.shp')
bolivia = world[world['NAME'] == 'Bolivia']
south_america = world[world['CONTINENT'] == 'South America']

# Crear GeoDataFrame para ubicaciones de plantas de energía
gdf_power_plants = gpd.GeoDataFrame(df_power_plants, geometry=gpd.points_from_xy(df_power_plants['lon'], df_power_plants['lat']))

# Crear un diccionario para mapear tipos de combustible a colores
fuel_colors = {
    'Hydro': 'blue',
    'Coal': 'black',
    'Natural Gas': 'orange',
    'Oil': 'red',
    'Wind': 'green',
    'Solar': 'yellow',
    'Bioenergy': 'brown',
}

# Mapear los tipos de combustible a colores en el DataFrame
df_power_plants['MarkerColor'] = df_power_plants['Fueltype'].map(fuel_colors)

# Crear un diccionario para mapear los valores de Un_ a colores
transmission_colors = {
    69: 'red',
    115: 'blue',
    230: 'green'
}

# Crear el mapa
fig, ax = plt.subplots(figsize=(12, 12))

# Plotear el mapa de Bolivia con los departamentos (más transparencia en el borde de los municipios)
bolivia_departamentos.plot(color=bolivia_departamentos['color'], 
                           ax=ax, 
                           edgecolor='black',    # El color de borde
                           alpha=0.1,            # Aumentar la transparencia
                           linewidth=0.1)        # Hacer las líneas más delgadas

# Resaltar los bordes externos de las zonas
for zona, color in [('Norte', 'black'), ('Central', 'black'), ('Oriental', 'black'), ('Sur', 'black')]:
    zona_departamentos = bolivia_departamentos[bolivia_departamentos['zona'] == zona]
    # Union de los departamentos dentro de cada zona para resaltar bordes externos
    zona_union = zona_departamentos.dissolve(by='zona')
    zona_union.boundary.plot(ax=ax, edgecolor=color, linewidth=1, alpha=1)  # Borde más visible

# Parámetro para modificar el grosor de las líneas de transmisión en el mapa
transmission_line_width = 0.3  # Hacer las líneas de transmisión más finas

# Plotear las líneas de transmisión con colores basados en valores de Un_
for value, color in transmission_colors.items():
    subset = gdf_transmission_lines[gdf_transmission_lines['Un_'] == value]
    subset.plot(ax=ax, color=color, linewidth=transmission_line_width, label=f'Transmission Lines {value} kV')

# Plotear las plantas de energía con tamaño proporcional a su capacidad
for fuel_type, color in fuel_colors.items():
    filtered_df = gdf_power_plants[gdf_power_plants['Fueltype'] == fuel_type]
    filtered_df.plot(ax=ax, markersize=filtered_df['Capacity'] * 1, color=color, alpha=0.8, label=f'Power Plants ({fuel_type})')

# Añadir título y etiquetas
ax.set_title('Power Plants and Transmission Lines in Bolivia', fontsize=16, fontweight='bold')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

# Crear marcadores personalizados para la leyenda de plantas de energía
legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=f'Power Plants ({fuel_type})') for fuel_type, color in fuel_colors.items()]

# Crear líneas personalizadas para la leyenda de líneas de transmisión (mantener grosor original)
for value, color in transmission_colors.items():
    legend_elements.append(Line2D([0], [0], color=color, linewidth=2, label=f'Transmission Lines {value} kV'))  # Grosor fijo en la leyenda

# Añadir leyenda con marcadores personalizados y líneas de transmisión
ax.legend(handles=legend_elements, loc='upper right')

# Añadir la escala del mapa
scale_bar_length_km = 100
scale_bar_length_deg = scale_bar_length_km / 111
scale_bar_x_start = -69.5
scale_bar_y_start = -22.5
ax.plot([scale_bar_x_start, scale_bar_x_start + scale_bar_length_deg], [scale_bar_y_start, scale_bar_y_start], color='k', lw=5)
ax.text(scale_bar_x_start + scale_bar_length_deg / 2, scale_bar_y_start - 0.2, f'{scale_bar_length_km} km', ha='center', va='center', fontsize=10)

# Añadir la flecha de referencia norte
arrow_x_start = -69.5
arrow_y_start = -22.0
ax.annotate('', xy=(arrow_x_start, arrow_y_start + 1), xytext=(arrow_x_start, arrow_y_start),
            arrowprops=dict(facecolor='black', width=5, headwidth=15))
ax.text(arrow_x_start, arrow_y_start + 1.2, 'N', ha='center', va='center', fontsize=12)

# Añadir el mapa de ubicación de Bolivia en Sudamérica en la esquina inferior derecha
inset_ax = inset_axes(ax, width="20%", height="20%", loc='lower right', borderpad=0.2)
south_america.plot(ax=inset_ax, color='lightgrey', edgecolor='black', alpha=0.3)  # Transparencia para Sudamérica
bolivia.plot(ax=inset_ax, color='red', edgecolor='black', alpha=0.6)  # Transparencia para Bolivia en el inset
inset_ax.set_xticks([])
inset_ax.set_yticks([])
inset_ax.set_title('Location')

# Mostrar el gráfico
plt.show()



#%%

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point, MultiLineString

# Cargar el shapefile de Bolivia con los departamentos
bolivia_departamentos = gpd.read_file('C:/Users/navia/Documents/DISPASET MODELS/5.12ENERGY/Maps/Bolivia/municipios339.shp')

# Definir los colores para cada zona
colors = {
    'La Paz': '#4F5D75',   # Azul oscuro
    'Beni': '#4F5D75',     # Azul oscuro
    'Pando': '#4F5D75',    # Azul oscuro
    'Cochabamba': '#9A8C98',  # Gris violáceo
    'Oruro': '#9A8C98',       # Gris violáceo
    'Santa Cruz': '#BC6C25',  # Marrón tierra
    'Tarija': '#E5989B',      # Rosa pálido oscuro
    'Chuquisaca': '#E5989B',  # Rosa pálido oscuro
    'PotosÃ­': '#E5989B'       # Rosa pálido oscuro
}

# Asignar colores a los departamentos
bolivia_departamentos['color'] = bolivia_departamentos['departamen'].map(colors).fillna('#D3D3D3')

# Definir las zonas
zona_norte = ['La Paz', 'Beni', 'Pando']
zona_central = ['Cochabamba', 'Oruro']
zona_oriental = ['Santa Cruz']
zona_sur = ['PotosÃ­', 'Chuquisaca', 'Tarija']

# Crear una nueva columna para las zonas
bolivia_departamentos['zona'] = bolivia_departamentos['departamen'].apply(
    lambda x: 'Norte' if x in zona_norte else 
              'Central' if x in zona_central else 
              'Oriental' if x in zona_oriental else 
              'Sur' if x in zona_sur else None
)

# Cargar los datos de líneas de transmisión
gdf_transmission_lines = gpd.read_file('../Bolivia-DataBase-main/transmision_sin_20190416/transmision_sin_20190416.shp')

# Lista para almacenar las líneas de transmisión que cruzan zonas
cross_zone_lines = []

# Función para extraer los puntos de inicio y fin de una geometría
def get_start_end_points(geometry):
    if isinstance(geometry, LineString):
        coords = list(geometry.coords)
        return coords[0], coords[-1]  # Retorna el primer y último punto
    elif isinstance(geometry, MultiLineString):
        # Para geometrías de tipo MultiLineString, tomar la primera y última línea
        first_line = geometry.geoms[0]
        last_line = geometry.geoms[-1]
        return list(first_line.coords)[0], list(last_line.coords)[-1]
    return None, None

# Iterar sobre cada línea de transmisión
for idx, row in gdf_transmission_lines.iterrows():
    start_point, end_point = get_start_end_points(row.geometry)
    
    if start_point and end_point:
        # Crear puntos shapely
        start_geom = Point(start_point)
        end_geom = Point(end_point)

        # Crear GeoDataFrames temporales para los puntos de inicio y fin
        start_gdf = gpd.GeoDataFrame(geometry=[start_geom], crs=gdf_transmission_lines.crs)
        end_gdf = gpd.GeoDataFrame(geometry=[end_geom], crs=gdf_transmission_lines.crs)

        # Determinar en qué departamento (zona) están los puntos de inicio y fin usando intersección espacial
        start_zone = gpd.sjoin(start_gdf, bolivia_departamentos, how="left", predicate="within")['zona'].values[0]
        end_zone = gpd.sjoin(end_gdf, bolivia_departamentos, how="left", predicate="within")['zona'].values[0]

        # Si las zonas de inicio y fin son diferentes, añadir la línea recta entre los puntos
        if start_zone != end_zone:
            line = LineString([start_point, end_point])  # Crear una línea recta entre los dos puntos
            cross_zone_lines.append(line)  # Almacenar la línea para plotearla después

# Crear un nuevo GeoDataFrame con las líneas que cruzan zonas
cross_zone_gdf = gpd.GeoDataFrame(geometry=cross_zone_lines, crs=gdf_transmission_lines.crs)

# Ajustar el grosor y la transparencia de las líneas de transmisión
fig, ax = plt.subplots(figsize=(12, 12))

# Plotear el mapa de Bolivia con transparencia
bolivia_departamentos.plot(color=bolivia_departamentos['color'], ax=ax, edgecolor='black', alpha=0.2, linewidth=0.5)

# Resaltar los bordes externos de las zonas
for zona, color in [('Norte', 'black'), ('Central', 'black'), ('Oriental', 'black'), ('Sur', 'black')]:
    zona_departamentos = bolivia_departamentos[bolivia_departamentos['zona'] == zona]
    # Unión de los departamentos dentro de cada zona para resaltar bordes externos
    zona_union = zona_departamentos.dissolve(by='zona')
    zona_union.boundary.plot(ax=ax, edgecolor=color, linewidth=1, alpha=1)  # Borde más visible

# Plotear las líneas de transmisión que cruzan zonas con grosor y transparencia ajustados
cross_zone_gdf.plot(ax=ax, color='purple', linewidth=2, alpha=0.6)

# Añadir título, etiquetas y otros detalles del mapa
ax.set_title('Transmission Lines Crossing Different Zones in Bolivia', fontsize=16, fontweight='bold')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

# Mostrar el gráfico
plt.show()


#%%
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.lines import Line2D
from shapely.geometry import Point, LineString, MultiLineString

# Cargar el shapefile de Bolivia con los departamentos
bolivia_departamentos = gpd.read_file('C:/Users/navia/Documents/DISPASET MODELS/5.12ENERGY/Maps/Bolivia/municipios339.shp')

# Definir los colores serios para cada zona
colors = {
    'La Paz': '#4F5D75',   # Azul oscuro
    'Beni': '#4F5D75',     # Azul oscuro
    'Pando': '#4F5D75',    # Azul oscuro
    'Cochabamba': 'blue',  # Gris violáceo
    'Oruro': 'blue',       # Gris violáceo
    'Santa Cruz': '#BC6C25',  # Marrón tierra
    'Tarija': '#E5989B',      # Rosa pálido oscuro
    'Chuquisaca': '#E5989B',  # Rosa pálido oscuro
    'PotosÃ­': '#E5989B'       # Rosa pálido oscuro
}

# Lidiar con valores NaN asignándoles un color por defecto (gris claro)
bolivia_departamentos['color'] = bolivia_departamentos['departamen'].map(colors).fillna('#D3D3D3')

# Definir las zonas
zona_norte = ['La Paz', 'Beni', 'Pando']
zona_central = ['Cochabamba', 'Oruro']
zona_oriental = ['Santa Cruz']
zona_sur = ['PotosÃ­', 'Chuquisaca', 'Tarija']

# Crear una nueva columna para las zonas
bolivia_departamentos['zona'] = bolivia_departamentos['departamen'].apply(
    lambda x: 'Norte' if x in zona_norte else 
              'Central' if x in zona_central else 
              'Oriental' if x in zona_oriental else 
              'Sur' if x in zona_sur else None
)

# Cargar los datos de líneas de transmisión y plantas de energía
df_power_plants = pd.read_csv('custom_powerplants.csv')
gdf_transmission_lines = gpd.read_file('../Bolivia-DataBase-main/transmision_sin_20190416/transmision_sin_20190416.shp')

# Cargar el shapefile de países para filtrar Bolivia
world = gpd.read_file('C:/Users/navia/Documents/DISPASET MODELS/5.12ENERGY/Maps/Countries/ne_10m_admin_0_countries.shp')
bolivia = world[world['NAME'] == 'Bolivia']
south_america = world[world['CONTINENT'] == 'South America']

# Crear GeoDataFrame para ubicaciones de plantas de energía
gdf_power_plants = gpd.GeoDataFrame(df_power_plants, geometry=gpd.points_from_xy(df_power_plants['lon'], df_power_plants['lat']))

# Crear un diccionario para mapear tipos de combustible a colores
fuel_colors = {
    'Hydro': 'blue',
    'Coal': 'black',
    'Natural Gas': 'orange',
    'Oil': 'red',
    'Wind': 'green',
    'Solar': 'yellow',
    'Bioenergy': 'brown',
}

# Mapear los tipos de combustible a colores en el DataFrame
df_power_plants['MarkerColor'] = df_power_plants['Fueltype'].map(fuel_colors)

# Crear un diccionario para mapear los valores de Un_ a colores
transmission_colors = {
    69: 'red',
    115: 'blue',
    230: 'green'
}

# Filtrar las líneas de transmisión que conectan dos zonas diferentes
cross_zone_lines = []

for idx, row in gdf_transmission_lines.iterrows():
    geometry = row.geometry
    
    if isinstance(geometry, LineString):
        start_point, end_point = geometry.coords[0], geometry.coords[-1]
    elif isinstance(geometry, MultiLineString):
        start_point = geometry.geoms[0].coords[0]
        end_point = geometry.geoms[-1].coords[-1]
    
    # Crear puntos shapely
    start_geom = Point(start_point)
    end_geom = Point(end_point)

    # Crear GeoDataFrames temporales para los puntos de inicio y fin
    start_gdf = gpd.GeoDataFrame(geometry=[start_geom], crs=gdf_transmission_lines.crs)
    end_gdf = gpd.GeoDataFrame(geometry=[end_geom], crs=gdf_transmission_lines.crs)

    # Determinar en qué departamento (zona) están los puntos de inicio y fin usando intersección espacial
    start_zone = gpd.sjoin(start_gdf, bolivia_departamentos, how="left", predicate="within")['zona'].values[0]
    end_zone = gpd.sjoin(end_gdf, bolivia_departamentos, how="left", predicate="within")['zona'].values[0]

    # Si las zonas de inicio y fin son diferentes, conservar la geometría original de la línea
    if start_zone != end_zone:
        cross_zone_lines.append(row)

# Convertir las líneas que cruzan zonas en un GeoDataFrame
cross_zone_gdf = gpd.GeoDataFrame(cross_zone_lines, crs=gdf_transmission_lines.crs)

# Crear el mapa
fig, ax = plt.subplots(figsize=(12, 12))

# Plotear el mapa de Bolivia con los departamentos (más transparencia)
bolivia_departamentos.plot(color=bolivia_departamentos['color'], ax=ax, edgecolor='black', alpha=0.2, linewidth=0.5)  # Menos prominente

# Resaltar los bordes externos de las zonas
for zona, color in [('Norte', 'black'), ('Central', 'black'), ('Oriental', 'black'), ('Sur', 'black')]:
    zona_departamentos = bolivia_departamentos[bolivia_departamentos['zona'] == zona]
    # Unión de los departamentos dentro de cada zona para resaltar bordes externos
    zona_union = zona_departamentos.dissolve(by='zona')
    zona_union.boundary.plot(ax=ax, edgecolor=color, linewidth=1, alpha=1)  # Borde más visible

# Parámetro para modificar el grosor de las líneas de transmisión en el mapa
transmission_line_width = 1  # Ajusta este valor al grosor deseado

# Plotear solo las líneas de transmisión que cruzan zonas diferentes
for value, color in transmission_colors.items():
    subset = cross_zone_gdf[cross_zone_gdf['Un_'] == value]
    subset.plot(ax=ax, color=color, linewidth=transmission_line_width, label=f'Transmission Lines {value} kV')

# Plotear las plantas de energía con tamaño proporcional a su capacidad
for fuel_type, color in fuel_colors.items():
    filtered_df = gdf_power_plants[gdf_power_plants['Fueltype'] == fuel_type]
    filtered_df.plot(ax=ax, markersize=filtered_df['Capacity'] * 1, color=color, alpha=0.8, label=f'Power Plants ({fuel_type})')

# Añadir título y etiquetas
ax.set_title('Power Plants and Transmission Lines in Bolivia', fontsize=16, fontweight='bold')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

# Crear marcadores personalizados para la leyenda de plantas de energía
legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=f'Power Plants ({fuel_type})') for fuel_type, color in fuel_colors.items()]

# Crear líneas personalizadas para la leyenda de líneas de transmisión (mantener grosor original)
for value, color in transmission_colors.items():
    legend_elements.append(Line2D([0], [0], color=color, linewidth=2, label=f'Transmission Lines {value} kV'))  # Grosor fijo en la leyenda

# Añadir leyenda con marcadores personalizados y líneas de transmisión
ax.legend(handles=legend_elements, loc='upper right')

# Añadir la escala del mapa
scale_bar_length_km = 100
scale_bar_length_deg = scale_bar_length_km / 111
scale_bar_x_start = -69.5
scale_bar_y_start = -22.5
ax.plot([scale_bar_x_start, scale_bar_x_start + scale_bar_length_deg], [scale_bar_y_start, scale_bar_y_start], color='k', lw=5)
ax.text(scale_bar_x_start + scale_bar_length_deg / 2, scale_bar_y_start - 0.2, f'{scale_bar_length_km} km', ha='center', va='center', fontsize=10)

# Añadir la flecha de referencia norte
arrow_x_start = -69.5
arrow_y_start = -22.0
ax.annotate('', xy=(arrow_x_start, arrow_y_start + 1), xytext=(arrow_x_start, arrow_y_start),
            arrowprops=dict(facecolor='black', width=5, headwidth=15))
ax.text(arrow_x_start, arrow_y_start + 1.2, 'N', ha='center', va='center', fontsize=12)

# Añadir el mapa de ubicación de Bolivia en Sudamérica en la esquina inferior derecha
inset_ax = inset_axes(ax, width="20%", height="20%", loc='lower right', borderpad=0.2)
south_america.plot(ax=inset_ax, color='lightgrey', edgecolor='black', alpha=0.3)  # Transparencia para Sudamérica
bolivia.plot(ax=inset_ax, color='red', edgecolor='black', alpha=0.6)  # Transparencia para Bolivia en el inset
inset_ax.set_xticks([])
inset_ax.set_yticks([])
inset_ax.set_title('Location')

# Mostrar el gráfico
plt.show()




#%%

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.lines import Line2D
from shapely.geometry import Point, LineString, MultiLineString
import numpy as np

# Función para convertir coordenadas de grados decimales a grados, minutos y segundos (DMS)
def decimal_to_dms(decimal_degrees):
    degrees = int(decimal_degrees)
    minutes = int((abs(decimal_degrees) - abs(degrees)) * 60)
    seconds = (abs(decimal_degrees) - abs(degrees) - minutes / 60) * 3600
    return f"{degrees}°{minutes}'{seconds:.2f}\""

# Cargar el shapefile de Bolivia con los departamentos
bolivia_departamentos = gpd.read_file('C:/Users/navia/Documents/DISPASET MODELS/5.12ENERGY/Maps/Bolivia/municipios339.shp')

# Definir los colores para cada zona
colors = {
    'La Paz': '#4F5D75',   # Azul oscuro
    'Beni': '#4F5D75',     # Azul oscuro
    'Pando': '#4F5D75',    # Azul oscuro
    'Cochabamba': 'blue',  # Gris violáceo
    'Oruro': 'blue',       # Gris violáceo
    'Santa Cruz': '#BC6C25',  # Marrón tierra
    'Tarija': '#E5989B',      # Rosa pálido oscuro
    'Chuquisaca': '#E5989B',  # Rosa pálido oscuro
    'PotosÃ­': '#E5989B'       # Rosa pálido oscuro
}

# Lidiar con valores NaN asignándoles un color por defecto (gris claro)
bolivia_departamentos['color'] = bolivia_departamentos['departamen'].map(colors).fillna('#D3D3D3')

# Definir las zonas
zona_norte = ['La Paz', 'Beni', 'Pando']
zona_central = ['Cochabamba', 'Oruro']
zona_oriental = ['Santa Cruz']
zona_sur = ['PotosÃ­', 'Chuquisaca', 'Tarija']

# Crear una nueva columna para las zonas
bolivia_departamentos['zona'] = bolivia_departamentos['departamen'].apply(
    lambda x: 'Norte' if x in zona_norte else 
              'Central' if x in zona_central else 
              'Oriental' if x in zona_oriental else 
              'Sur' if x in zona_sur else None
)

# Cargar los datos de líneas de transmisión y plantas de energía
df_power_plants = pd.read_csv('custom_powerplants.csv')
gdf_transmission_lines = gpd.read_file('../Bolivia-DataBase-main/transmision_sin_20190416/transmision_sin_20190416.shp')

# Cargar el shapefile de países para filtrar Bolivia
world = gpd.read_file('C:/Users/navia/Documents/DISPASET MODELS/5.12ENERGY/Maps/Countries/ne_10m_admin_0_countries.shp')
bolivia = world[world['NAME'] == 'Bolivia']
south_america = world[world['CONTINENT'] == 'South America']

# Crear GeoDataFrame para ubicaciones de plantas de energía
gdf_power_plants = gpd.GeoDataFrame(df_power_plants, geometry=gpd.points_from_xy(df_power_plants['lon'], df_power_plants['lat']))

# Crear un diccionario para mapear tipos de combustible a colores
fuel_colors = {
    'Hydro': 'blue',
    'Coal': 'black',
    'Natural Gas': 'orange',
    'Oil': 'red',
    'Wind': 'green',
    'Solar': 'yellow',
    'Bioenergy': 'brown',
}

# Mapear los tipos de combustible a colores en el DataFrame
df_power_plants['MarkerColor'] = df_power_plants['Fueltype'].map(fuel_colors)

# Crear un diccionario para mapear los valores de Un_ a colores
transmission_colors = {
    69: 'red',
    115: 'blue',
    230: 'green'
}

# Crear el mapa
fig, ax = plt.subplots(figsize=(12, 12))

# Plotear el mapa de Bolivia con los departamentos (más transparencia)
bolivia_departamentos.plot(color=bolivia_departamentos['color'], ax=ax, edgecolor='black', alpha=0.2, linewidth=0.5)  # Menos prominente

# Resaltar los bordes externos de las zonas
for zona, color in [('Norte', 'black'), ('Central', 'black'), ('Oriental', 'black'), ('Sur', 'black')]:
    zona_departamentos = bolivia_departamentos[bolivia_departamentos['zona'] == zona]
    # Unión de los departamentos dentro de cada zona para resaltar bordes externos
    zona_union = zona_departamentos.dissolve(by='zona')
    zona_union.boundary.plot(ax=ax, edgecolor=color, linewidth=1, alpha=1)

# Plotear solo las líneas de transmisión que cruzan zonas diferentes
for value, color in transmission_colors.items():
    subset = gdf_transmission_lines[gdf_transmission_lines['Un_'] == value]  # Asegúrate de que este nombre sea correcto
    subset.plot(ax=ax, color=color, linewidth=1, label=f'Transmission Lines {value} kV')

# Plotear las plantas de energía con tamaño proporcional a su capacidad
for fuel_type, color in fuel_colors.items():
    filtered_df = gdf_power_plants[gdf_power_plants['Fueltype'] == fuel_type]
    filtered_df.plot(ax=ax, markersize=filtered_df['Capacity'] * 1, color=color, alpha=0.8, label=f'Power Plants ({fuel_type})')

# Añadir título y etiquetas
ax.set_title('Power Plants and Transmission Lines in Bolivia', fontsize=16, fontweight='bold')

# Convertir los ticks del eje X e Y a formato DMS (grados, minutos, segundos)
xticks = ax.get_xticks()
yticks = ax.get_yticks()
ax.set_xticklabels([decimal_to_dms(tick) for tick in xticks])
ax.set_yticklabels([decimal_to_dms(tick) for tick in yticks])

# Ajustar el zoom al área de interés, definiendo los límites (xlim y ylim)
# Ejemplo: acercar a la región central de Bolivia (ajusta las coordenadas según tu interés)
ax.set_xlim([-70, -61])  # Ajusta estos valores para longitud
ax.set_ylim([-22.5, -13.5])  # Ajusta estos valores para latitud

# Leyenda
legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=f'Power Plants ({fuel_type})') for fuel_type, color in fuel_colors.items()]

# Líneas de transmisión en la leyenda
for value, color in transmission_colors.items():
    legend_elements.append(Line2D([0], [0], color=color, linewidth=2, label=f'Transmission Lines {value} kV'))

ax.legend(handles=legend_elements, loc='upper right')

# Escala del mapa
scale_bar_length_km = 100  # Ajustar la longitud de la barra de escala
scale_bar_length_deg = scale_bar_length_km / 111  # Conversión a grados
scale_bar_x_start = -68.5  # Ajusta la posición en el eje X
scale_bar_y_start = -19.5  # Ajusta la posición en el eje Y
ax.plot([scale_bar_x_start, scale_bar_x_start + scale_bar_length_deg], [scale_bar_y_start, scale_bar_y_start], color='k', lw=5)
ax.text(scale_bar_x_start + scale_bar_length_deg / 2, scale_bar_y_start - 0.1, f'{scale_bar_length_km} km', ha='center', va='center', fontsize=10)

# Flecha norte
arrow_x_start = -68.8  # Ajustar la posición de la flecha
arrow_y_start = -16.5
ax.annotate('', xy=(arrow_x_start, arrow_y_start + 1), xytext=(arrow_x_start, arrow_y_start),
            arrowprops=dict(facecolor='black', width=5, headwidth=15))
ax.text(arrow_x_start, arrow_y_start + 1.2, 'N', ha='center', va='center', fontsize=12)

# Inset: Mapa de ubicación de Bolivia
inset_ax = inset_axes(ax, width="20%", height="20%", loc='lower right', borderpad=0.2)
south_america.plot(ax=inset_ax, color='lightgrey', edgecolor='black', alpha=0.3)
bolivia.plot(ax=inset_ax, color='red', edgecolor='black', alpha=0.6)
inset_ax.set_xticks([])
inset_ax.set_yticks([])
inset_ax.set_title('Location')

# Ajustar los márgenes para que el mapa ocupe más espacio
plt.tight_layout()

# Mostrar el gráfico
plt.show()


#%%
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.lines import Line2D
from rasterio.plot import show
import rasterio
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Cargar el shapefile de Bolivia con los departamentos
bolivia_departamentos = gpd.read_file('C:/Users/navia/Documents/DISPASET MODELS/5.12ENERGY/Maps/Bolivia/municipios339.shp')

# Definir los colores serios para cada zona
colors = {
    'La Paz': '#4F5D75',   # Azul oscuro
    'Beni': '#4F5D75',     # Azul oscuro
    'Pando': '#4F5D75',    # Azul oscuro
    'Cochabamba': 'blue',  # Gris violáceo
    'Oruro': 'blue',       # Gris violáceo
    'Santa Cruz': '#BC6C25',  # Marrón tierra
    'Tarija': '#E5989B',      # Rosa pálido oscuro
    'Chuquisaca': '#E5989B',  # Rosa pálido oscuro
    'PotosÃ­': '#E5989B'       # Rosa pálido oscuro
}

# Lidiar con valores NaN asignándoles un color por defecto (gris claro)
bolivia_departamentos['color'] = bolivia_departamentos['departamen'].map(colors).fillna('#D3D3D3')

# Definir las zonas
zona_norte = ['La Paz', 'Beni', 'Pando']
zona_central = ['Cochabamba', 'Oruro']
zona_oriental = ['Santa Cruz']
zona_sur = ['PotosÃ­', 'Chuquisaca', 'Tarija']

# Crear una nueva columna para las zonas
bolivia_departamentos['zona'] = bolivia_departamentos['departamen'].apply(
    lambda x: 'Norte' if x in zona_norte else 
              'Central' if x in zona_central else 
              'Oriental' if x in zona_oriental else 
              'Sur' if x in zona_sur else None
)

# Cargar los datos de líneas de transmisión y plantas de energía
df_power_plants = pd.read_csv('custom_powerplants.csv')
gdf_transmission_lines = gpd.read_file('../Bolivia-DataBase-main/transmision_sin_20190416/transmision_sin_20190416.shp')

# Cargar el shapefile de países para filtrar Bolivia
world = gpd.read_file('C:/Users/navia/Documents/DISPASET MODELS/5.12ENERGY/Maps/Countries/ne_10m_admin_0_countries.shp')
bolivia = world[world['NAME'] == 'Bolivia']
south_america = world[world['CONTINENT'] == 'South America']

# Crear GeoDataFrame para ubicaciones de plantas de energía
gdf_power_plants = gpd.GeoDataFrame(df_power_plants, geometry=gpd.points_from_xy(df_power_plants['lon'], df_power_plants['lat']))

# Crear un diccionario para mapear tipos de combustible a colores
fuel_colors = {
    'Hydro': 'blue',
    'Coal': 'black',
    'Natural Gas': 'orange',
    'Oil': 'red',
    'Wind': 'green',
    'Solar': 'yellow',
    'Bioenergy': 'brown',
}

# Mapear los tipos de combustible a colores en el DataFrame
df_power_plants['MarkerColor'] = df_power_plants['Fueltype'].map(fuel_colors)

# Crear un diccionario para mapear los valores de Un_ a colores
transmission_colors = {
    69: 'red',
    115: 'blue',
    230: 'green'
}

# Crear el mapa
fig, ax = plt.subplots(figsize=(12, 12))

# Plotear el mapa de Bolivia con los departamentos
bolivia_departamentos.plot(color=bolivia_departamentos['color'], ax=ax, edgecolor='black', alpha=0.2, linewidth=0.5)

# Resaltar los bordes externos de las zonas
for zona, color in [('Norte', 'black'), ('Central', 'black'), ('Oriental', 'black'), ('Sur', 'black')]:
    zona_departamentos = bolivia_departamentos[bolivia_departamentos['zona'] == zona]
    zona_union = zona_departamentos.dissolve(by='zona')
    zona_union.boundary.plot(ax=ax, edgecolor=color, linewidth=1, alpha=1)

# Parámetro para modificar el grosor de las líneas de transmisión en el mapa
transmission_line_width = 1

# Plotear las líneas de transmisión con colores basados en valores de Un_
for value, color in transmission_colors.items():
    subset = gdf_transmission_lines[gdf_transmission_lines['Un_'] == value]
    subset.plot(ax=ax, color=color, linewidth=transmission_line_width, label=f'Transmission Lines {value} kV')

# Plotear las plantas de energía con tamaño proporcional a su capacidad
for fuel_type, color in fuel_colors.items():
    filtered_df = gdf_power_plants[gdf_power_plants['Fueltype'] == fuel_type]
    filtered_df.plot(ax=ax, markersize=filtered_df['Capacity'] * 1, color=color, alpha=0.8, label=f'Power Plants ({fuel_type})')

# Añadir título y etiquetas
ax.set_title('Power Plants and Transmission Lines in Bolivia', fontsize=16, fontweight='bold')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

# Crear marcadores personalizados para la leyenda de plantas de energía
legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=f'Power Plants ({fuel_type})') for fuel_type, color in fuel_colors.items()]

# Crear líneas personalizadas para la leyenda de líneas de transmisión
for value, color in transmission_colors.items():
    legend_elements.append(Line2D([0], [0], color=color, linewidth=2, label=f'Transmission Lines {value} kV'))

# Añadir leyenda con marcadores personalizados y líneas de transmisión
ax.legend(handles=legend_elements, loc='upper right')

# Añadir la escala del mapa
scale_bar_length_km = 100
scale_bar_length_deg = scale_bar_length_km / 111
scale_bar_x_start = -69.5
scale_bar_y_start = -22.5
ax.plot([scale_bar_x_start, scale_bar_x_start + scale_bar_length_deg], [scale_bar_y_start, scale_bar_y_start], color='k', lw=5)
ax.text(scale_bar_x_start + scale_bar_length_deg / 2, scale_bar_y_start - 0.2, f'{scale_bar_length_km} km', ha='center', va='center', fontsize=10)

# Añadir la flecha de referencia norte
arrow_x_start = -69.5
arrow_y_start = -22.0
ax.annotate('', xy=(arrow_x_start, arrow_y_start + 1), xytext=(arrow_x_start, arrow_y_start),
            arrowprops=dict(facecolor='black', width=5, headwidth=15))
ax.text(arrow_x_start, arrow_y_start + 1.2, 'N', ha='center', va='center', fontsize=12)

# Añadir el mapa de ubicación de Bolivia en Sudamérica en la esquina inferior derecha
inset_ax = inset_axes(ax, width="20%", height="20%", loc='lower right', borderpad=0.2)
south_america.plot(ax=inset_ax, color='lightgrey', edgecolor='black', alpha=0.3)
bolivia.plot(ax=inset_ax, color='red', edgecolor='black', alpha=0.6)
inset_ax.set_xticks([])
inset_ax.set_yticks([])
inset_ax.set_title('Location')

# Leer y mostrar el mapa de radiación solar
with rasterio.open('C:/Users/navia/Documents/DISPASET MODELS/5.12ENERGY/Maps/Solar/Bolivia_GISdata_LTAy_AvgDailyTotals_GlobalSolarAtlas-v2_GEOTIFF/DNI.tif') as src:
    show(src, ax=ax, cmap='autumn', alpha=0.5)  # Usar la escala de colores de otoño (amarillo a rojo)

# Mostrar el gráfico
plt.show()

#%%

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.lines import Line2D
from rasterio.plot import show
import rasterio
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Cargar el shapefile de Bolivia con los departamentos
bolivia_departamentos = gpd.read_file('C:/Users/navia/Documents/DISPASET MODELS/5.12ENERGY/Maps/Bolivia/municipios339.shp')

# Definir los colores serios para cada zona
colors = {
    'La Paz': '#4F5D75',   # Azul oscuro
    'Beni': '#4F5D75',     # Azul oscuro
    'Pando': '#4F5D75',    # Azul oscuro
    'Cochabamba': 'blue',  # Gris violáceo
    'Oruro': 'blue',       # Gris violáceo
    'Santa Cruz': '#BC6C25',  # Marrón tierra
    'Tarija': '#E5989B',      # Rosa pálido oscuro
    'Chuquisaca': '#E5989B',  # Rosa pálido oscuro
    'PotosÃ­': '#E5989B'       # Rosa pálido oscuro
}

# Lidiar con valores NaN asignándoles un color por defecto (gris claro)
bolivia_departamentos['color'] = bolivia_departamentos['departamen'].map(colors).fillna('#D3D3D3')

# Definir las zonas
zona_norte = ['La Paz', 'Beni', 'Pando']
zona_central = ['Cochabamba', 'Oruro']
zona_oriental = ['Santa Cruz']
zona_sur = ['PotosÃ­', 'Chuquisaca', 'Tarija']

# Crear una nueva columna para las zonas
bolivia_departamentos['zona'] = bolivia_departamentos['departamen'].apply(
    lambda x: 'Norte' if x in zona_norte else 
              'Central' if x in zona_central else 
              'Oriental' if x in zona_oriental else 
              'Sur' if x in zona_sur else None
)

# Cargar los datos de líneas de transmisión y plantas de energía
df_power_plants = pd.read_csv('custom_powerplants.csv')
gdf_transmission_lines = gpd.read_file('../Bolivia-DataBase-main/transmision_sin_20190416/transmision_sin_20190416.shp')

# Cargar el shapefile de países para filtrar Bolivia
world = gpd.read_file('C:/Users/navia/Documents/DISPASET MODELS/5.12ENERGY/Maps/Countries/ne_10m_admin_0_countries.shp')
bolivia = world[world['NAME'] == 'Bolivia']
south_america = world[world['CONTINENT'] == 'South America']

# Crear GeoDataFrame para ubicaciones de plantas de energía
gdf_power_plants = gpd.GeoDataFrame(df_power_plants, geometry=gpd.points_from_xy(df_power_plants['lon'], df_power_plants['lat']))

# Crear un diccionario para mapear tipos de combustible a colores
fuel_colors = {
    'Hydro': 'blue',
    'Coal': 'black',
    'Natural Gas': 'orange',
    'Oil': 'red',
    'Wind': 'green',
    'Solar': 'yellow',
    'Bioenergy': 'brown',
}

# Mapear los tipos de combustible a colores en el DataFrame
df_power_plants['MarkerColor'] = df_power_plants['Fueltype'].map(fuel_colors)

# Crear un diccionario para mapear los valores de Un_ a colores
transmission_colors = {
    69: 'red',
    115: 'blue',
    230: 'green'
}

# Crear el mapa
fig, ax = plt.subplots(figsize=(12, 12))

# Plotear el mapa de Bolivia con los departamentos
bolivia_departamentos.plot(color=bolivia_departamentos['color'], ax=ax, edgecolor='black', alpha=0.2, linewidth=0.5)

# Resaltar los bordes externos de las zonas
for zona, color in [('Norte', 'black'), ('Central', 'black'), ('Oriental', 'black'), ('Sur', 'black')]:
    zona_departamentos = bolivia_departamentos[bolivia_departamentos['zona'] == zona]
    zona_union = zona_departamentos.dissolve(by='zona')
    zona_union.boundary.plot(ax=ax, edgecolor=color, linewidth=1, alpha=1)

# Parámetro para modificar el grosor de las líneas de transmisión en el mapa
transmission_line_width = 1

# Plotear las líneas de transmisión con colores basados en valores de Un_
for value, color in transmission_colors.items():
    subset = gdf_transmission_lines[gdf_transmission_lines['Un_'] == value]
    subset.plot(ax=ax, color=color, linewidth=transmission_line_width, label=f'Transmission Lines {value} kV')

# Plotear las plantas de energía con tamaño proporcional a su capacidad
for fuel_type, color in fuel_colors.items():
    filtered_df = gdf_power_plants[gdf_power_plants['Fueltype'] == fuel_type]
    filtered_df.plot(ax=ax, markersize=filtered_df['Capacity'] * 1, color=color, alpha=0.8, label=f'Power Plants ({fuel_type})')

# Añadir título y etiquetas
ax.set_title('Solar Radiation in Bolivia', fontsize=16, fontweight='bold')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

# Crear marcadores personalizados para la leyenda de plantas de energía
legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=f'Power Plants ({fuel_type})') for fuel_type, color in fuel_colors.items()]

# Crear líneas personalizadas para la leyenda de líneas de transmisión
for value, color in transmission_colors.items():
    legend_elements.append(Line2D([0], [0], color=color, linewidth=2, label=f'Transmission Lines {value} kV'))

# Añadir leyenda con marcadores personalizados y líneas de transmisión
ax.legend(handles=legend_elements, loc='upper right')

# Añadir la escala del mapa
scale_bar_length_km = 100
scale_bar_length_deg = scale_bar_length_km / 111
scale_bar_x_start = -69.5
scale_bar_y_start = -22.5
ax.plot([scale_bar_x_start, scale_bar_x_start + scale_bar_length_deg], [scale_bar_y_start, scale_bar_y_start], color='k', lw=5)
ax.text(scale_bar_x_start + scale_bar_length_deg / 2, scale_bar_y_start - 0.2, f'{scale_bar_length_km} km', ha='center', va='center', fontsize=10)

# Añadir la flecha de referencia norte
arrow_x_start = -69.5
arrow_y_start = -22.0
ax.annotate('', xy=(arrow_x_start, arrow_y_start + 1), xytext=(arrow_x_start, arrow_y_start),
            arrowprops=dict(facecolor='black', width=5, headwidth=15))
ax.text(arrow_x_start, arrow_y_start + 1.2, 'N', ha='center', va='center', fontsize=12)

# Añadir el mapa de ubicación de Bolivia en Sudamérica en la esquina inferior derecha
inset_ax = inset_axes(ax, width="20%", height="20%", loc='lower right', borderpad=0.2)
south_america.plot(ax=inset_ax, color='lightgrey', edgecolor='black', alpha=0.3)
bolivia.plot(ax=inset_ax, color='red', edgecolor='black', alpha=0.6)
inset_ax.set_xticks([])
inset_ax.set_yticks([])
inset_ax.set_title('Location')

# # Leer y mostrar el mapa de radiación solar
# with rasterio.open('C:/Users/navia/Documents/DISPASET MODELS/5.12ENERGY/Maps/Solar/Bolivia_GISdata_LTAy_AvgDailyTotals_GlobalSolarAtlas-v2_GEOTIFF/DNI.tif') as src:
#     # Crear un mapa de colores 'viridis'
#     cmap = plt.get_cmap('viridis')
#     show(src, ax=ax, cmap=cmap, alpha=0.5)  # Aplicar la escala de colores 'viridis'

# with rasterio.open('C:/Users/navia/Documents/DISPASET MODELS/5.12ENERGY/Maps/Solar/Bolivia_GISdata_LTAy_AvgDailyTotals_GlobalSolarAtlas-v2_GEOTIFF/DNI.tif') as src:
#     # Crear un mapa de colores 'plasma'
#     cmap = plt.get_cmap('plasma')  # Puedes probar también 'inferno' o 'magma'
#     # fig, ax = plt.subplots(figsize=(10, 10))
#     show(src, ax=ax, cmap=cmap, alpha=0.5)  # Aplicar la escala de colores 'plasma'

# Definir una paleta de colores personalizada basada en el modelo proporcionado
colors1 = ["#ffffcc", "#ffeda0", "#fed976", "#feb24c", "#fd8d3c", "#fc4e2a", "#e31a1c", "#bd0026", "#800026"]
cmap = LinearSegmentedColormap.from_list("custom_cmap", colors1, N=256)

with rasterio.open('C:/Users/navia/Documents/DISPASET MODELS/5.12ENERGY/Maps/Solar/Bolivia_GISdata_LTAy_AvgDailyTotals_GlobalSolarAtlas-v2_GEOTIFF/DNI.tif') as src:
    # fig, ax = plt.subplots(figsize=(10, 10))
    show(src, ax=ax, cmap=cmap, alpha=0.5)  # Aplicar la escala de colores personalizada
    # plt.title('Mapa de Radiación Solar en Bolivia')
    plt.colorbar(ax.collections[0], ax=ax, orientation='horizontal', label='kWh/m²/day')
    # plt.show()

# Mostrar el gráfico
plt.show()

