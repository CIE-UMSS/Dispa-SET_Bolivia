# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 11:03:38 2025

@author: navia
"""

import matplotlib.pyplot as plt
import networkx as nx

def plot_demand_flow():
    G = nx.DiGraph()
    
    # Nombres de los sectores energéticos
    sectors = ["Residential", "Public Lighting", "Commercial", "Industrial",
               "Transport", "Agriculture", "Mining", "Fishing & Others", "Non-energy"]
    
    # Consumo energético relativo por sector (ejemplos hipotéticos, ajusta según datos reales)
    sector_consumption = {
        "Residential": 15,
        "Public Lighting": 5,
        "Commercial": 10,
        "Industrial": 20,
        "Transport": 25,
        "Agriculture": 5,
        "Mining": 5,
        "Fishing & Others": 2,
        "Non-energy": 1
    }
    
    # Normalización de tamaños de nodos basados en el consumo
    max_consumption = max(sector_consumption.values())
    min_consumption = min(sector_consumption.values())
    size_range = (1000, 3000)  # Rango de tamaños de nodos
    size_scaling = (size_range[1] - size_range[0]) / (max_consumption - min_consumption)
    
    # Asignación de tamaños y colores
    sector_colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral', 'lightpink', 
                     'lightskyblue', 'lightgray', 'lightseagreen', 'lightgoldenrodyellow']
    sector_sizes = {
        sector: size_range[0] + (consumption - min_consumption) * size_scaling
        for sector, consumption in sector_consumption.items()
    }
    
    # Agregar nodos de sectores energéticos
    for i, sector in enumerate(sectors):
        G.add_node(sector, color=sector_colors[i], size=sector_sizes[sector])
    
    # Nodo de demanda total
    G.add_node("Total Demand", color='red', size=2000)
    
    # Zonas geográficas
    zones = ["CE (Centro)", "OR (Oriente)", "NO (Norte)", "SU (Sur)"]
    for zone in zones:
        G.add_node(zone, color='orange', size=1500)
    
    # Modelos energéticos
    models = ["ESOM (PyPSA)", "UCOD"]
    
    # Colores y tamaños para los modelos energéticos
    model_colors = ['green', 'blue']
    model_sizes = [1800, 1600]
    
    for i, model in enumerate(models):
        G.add_node(model, color=model_colors[i], size=model_sizes[i])
    
    # Conexiones entre sectores energéticos y demanda total (flechas directas)
    for sector in sectors:
        G.add_edge(sector, "Total Demand", color='black', weight=2)
    
    # Conexiones entre zonas y modelos energéticos
    for zone in zones:
        for model in models:
            G.add_edge(zone, model, color='gray', weight=1)
    
    # Conexiones entre modelos energéticos y demanda total (directas)
    G.add_edge("ESOM (PyPSA)", "Total Demand", color='green', weight=3)
    G.add_edge("UCOD", "Total Demand", color='blue', weight=3)
    
    # Posiciones de los nodos (ajustando la posición de EnergyScope a la izquierda)
    pos = {
        # Posición de "Total Demand" en el centro
        "Total Demand": (0.5, 0.5),
        
        # Sectores energéticos distribuidos alrededor de la demanda total
        "Residential": (0.8, 0.8), "Public Lighting": (0.7, 0.9), "Commercial": (0.6, 1.0), "Industrial": (0.5, 1.1),
        "Transport": (0.3, 1.0), "Agriculture": (0.2, 0.9), "Mining": (0.1, 0.8), "Fishing & Others": (0.2, 0.7),
        "Non-energy": (0.3, 0.6),
        
        # Zonas geográficas al centro
        "CE (Centro)": (0.5, 0.6), "OR (Oriente)": (0.5, 0.5), "NO (Norte)": (0.5, 0.4), "SU (Sur)": (0.5, 0.3),
        
        # Modelos energéticos (EnergyScope y UCOD) ahora más a la izquierda
        "ESOM (PyPSA)": (0.1, 0.6), "UCOD": (0.1, 0.4),
    }
    
    # Colores y tamaños de los nodos
    colors = [G.nodes[node]['color'] for node in G.nodes]
    sizes = [G.nodes[node]['size'] for node in G.nodes]
    
    # Graficar el gráfico con las posiciones definidas
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_color=colors, node_size=sizes, edge_color='gray', font_size=10, arrows=True)
    
    # Agregar etiquetas de los ejes
    plt.xlabel("Modelo Energético: ENERGYSCOPE (eje x)", fontsize=12)
    plt.ylabel("Horizonte Temporal: Décadas (eje y)", fontsize=12)
    
    # Título
    plt.title("Flujo de Construcción de la Demanda Energética", fontsize=14)
    
    # Mostrar el gráfico
    plt.grid(True)
    plt.show()

# Generar el gráfico
plot_demand_flow()
