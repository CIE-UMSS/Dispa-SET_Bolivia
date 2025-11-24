# -*- coding: utf-8 -*-
"""
This script runs the Dispa-SET EU model with the 2016 data. The main steps are:
    - Load Dispa-SET
    - Load the config file for the EU model
    - build the mode
    - run the model
    - display and analyse the results

@author: Sylvain Quoilin
"""

# Add the root folder of Dispa-SET to the path so that the library can be loaded:
import os
import sys
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
    

sys.path.append(os.path.abspath('..'))

# Import Dispa-SET
import dispaset as ds


#SCRIPT TO BUILT, RUN AND READ SEVERAL SCENARIOS
scenarios = {}
#  
# sce_list = ['S0','S1','S2','S3','S4','S5','S6']
# sce_list = ['S0_SI','S1_SI','S2_SI','S3_SI','S4_SI','S5_SI','S6_SI']
# sce_list = ['S0_DI','S1_DI','S2_DI','S3_DI','S4_DI','S5_DI','S6_DI']
sce_list = [
            'S0','S0_DI','S0_SI',
            'S1','S1_DI','S1_SI',
            'S2','S2_DI','S2_SI',
            'S3','S3_DI','S3_SI',
            'S4','S4_DI','S4_SI',
            'S5','S5_DI','S5_SI'
            ]

for s in sce_list: 
    
    # Load the configuration file
    config_path = f'../ConfigFiles/Config_BOLIVIA_{s}.xlsx'
    config_name = f'config_{s}'
    scenarios[config_name] = ds.load_config(config_path)
    
    # # Limit the simulation period (for testing purposes, comment the line to run the whole year)
    # scenarios[config_name]['StartDate'] = (2026, 1, 1, 0, 0, 0)
    # scenarios[config_name]['StopDate'] = (2026, 7, 1, 0, 0, 0)
    
    # # Build the simulation environment:
    # # SimData_name = f'SimData_{s}'
    # # SimData_name = ds.build_simulation(scenarios[config_name])
    # SimData_name = f'SimData_{s}'
    # scenarios[SimData_name] = ds.build_simulation(scenarios[config_name],mts_plot=True,MTSTimeStep=24)
    
    # # # Solve using GAMS:
    # solve_name = f'solve_{s}'
    # scenarios[solve_name] = ds.solve_GAMS(scenarios[config_name]['SimulationDirectory'], scenarios[config_name]['GAMS_folder'])
    
    # Load the simulation results:
    sim_path = f'../Simulations/BOLIVIA_{s}'
    inputs_name = f'inputs_{s}'
    results_name = f'results_{s}'
    scenarios[inputs_name],scenarios[results_name] = ds.get_sim_results(path=sim_path,cache=False)
    
    inputs_MTS_name = f'inputs_MTS_{s}'
    results_MTS_name = f'results_MTS_{s}'
    scenarios[inputs_MTS_name],scenarios[results_MTS_name] = ds.get_sim_results(path=sim_path, cache=False, inputs_file='Inputs_MTS.p',results_file='Results_MTS.gdx')
    
    # # Generate country-specific plots
    # ds.plot_zone(scenarios[inputs_name],scenarios[results_name], rng=rng)
    # rng = pd.date_range('2026-01-01', '2026-01-07', freq='H')
    # for i in scenarios[config_name]['zones']:
    #     ds.plot_zone(scenarios[inputs_name],scenarios[results_name], z=i, rng=rng)
        
    # rng = pd.date_range('2030-01-01', '2030-12-31', freq='H')
    # for i in scenarios[config_name]['zones']:
    #     ds.plot_zone(scenarios[inputs_name],scenarios[results_name], z=i, rng=rng)
        
    # rng1 = pd.date_range('2030-01-01', '2030-12-31', freq='D')
    # for j in scenarios[config_name]['zones']:
    #     ds.plot_zone(scenarios[inputs_MTS_name],scenarios[results_MTS_name], z=j, rng=rng1)
    
    # # Bar plot with the installed capacities in all countries:
    # cap_name = f'cap_{s}'
    # scenarios[cap_name] = ds.plot_zone_capacities(scenarios[inputs_name],scenarios[results_name])
    
    # # Bar plot with installed storage capacity
    # sto_name = f'sto_{s}'
    # scenarios[sto_name] = ds.plot_tech_cap(scenarios[inputs_name])
    
    # # Violin plot for CO2 emissions
    # ds.plot_co2(scenarios[inputs_name],scenarios[results_name], figsize=(9, 6), width=0.9)
    
    # # Bar plot with the energy balances in all countries:
    # ds.plot_energy_zone_fuel(scenarios[inputs_name],scenarios[results_name], ds.get_indicators_powerplant(scenarios[inputs_name],scenarios[results_name]))
    
    # Analyse the results for each country and provide quantitative indicators:
    r_name = f'r_{s}'
    scenarios[r_name] = ds.get_result_analysis(scenarios[inputs_name],scenarios[results_name])
    
    # # Analyze power flow tracing
    # pft_name = f'pft_prct_{s}'
    # pft_prct_name = f'pft_prct_{s}'
    # scenarios[pft_name], scenarios[pft_prct_name] = ds.plot_power_flow_tracing_matrix(scenarios[inputs_name],scenarios[results_name], cmap="magma_r", figsize=(15, 10))
    
    # # Plot net flows on a map
    # ds.plot_net_flows_map(scenarios[inputs_name],scenarios[results_name], terrain=True, margin=3, bublesize=5000, figsize=(8, 7))
    
    # # Plot congestion in the interconnection lines on a map
    # ds.plot_line_congestion_map(scenarios[inputs_name],scenarios[results_name], terrain=True, margin=3, figsize=(9, 7), edge_width=3.5, bublesize=100)
    
    # # # Analyse the results of the dispatch and provide frequency security constraints:
    # freq_sec_const_name = f'freq_sec_const_{s}'
    # summary_name = f'Summary_{s}'
    # contingency_name = f'Contingency_{s}'
            
    # # Read the results of the frequency security constraints analisys   
    # idx = scenarios[inputs_name]['config']['idx']
    # scenarios[summary_name] = pd.read_csv(f'../Simulations/BOLIVIA_{s}/Summary.csv') 
    # scenarios[summary_name].set_index(idx, inplace=True)
    # scenarios[contingency_name] = pd.read_csv(f'../Simulations/BOLIVIA_{s}/Contingency.csv')  
    # scenarios[contingency_name].set_index(idx, inplace=True)
    # # scenarios[freq_sec_const_name] = pd.read_excel(f'../Simulations/BOLIVIA_{s}/frec_sec_const.xlsx', sheet_name=None)


# TODOS LOS VALORES UTILES PARA PLOTS DE INERCIA EN UN SOLO DATAFRAME
   
    # PARA ENCONTRAR TOTAL RENEWABLE GENERATION (TRG)
    df_new_plots_name = f'df_new_plots_{s}'
    scenarios[df_new_plots_name] = pd.DataFrame()
    vreunits = scenarios[inputs_name]['units']
    fuels=['SUN','WIN']
    vreunits = vreunits[vreunits.Fuel.isin(fuels)]
    vrelist = list(vreunits['Unit'])
    vreoutputpow = scenarios[results_name]['OutputPower']
    vreoutputpow = vreoutputpow.T
    vreoutputpow = vreoutputpow.reset_index()
    vreoutputpow.rename(columns = {'index':'Unit'}, inplace = True)
    vreoutputpow = vreoutputpow[vreoutputpow.Unit.isin(vrelist)]
    vreoutputpow = vreoutputpow.T
    vreoutputpow.columns = vreoutputpow.iloc[0]
    vreoutputpow = vreoutputpow[1:]
    scenarios[df_new_plots_name][f'TRG_{s}'] = vreoutputpow.sum(axis=1)
    
    
    # PARA ENCONTRAR TOTAL CONVENTIONAL GENERATION (TCONVG)
    convunits = scenarios[inputs_name]['units']
    fuels=['WAT','GAS','OIL','BIO']
    convunits = convunits[convunits.Fuel.isin(fuels)]
    convlist = list(convunits['Unit'])
    convoutputpow = scenarios[results_name]['OutputPower']
    convoutputpow = convoutputpow.T
    convoutputpow = convoutputpow.reset_index()
    convoutputpow.rename(columns = {'index':'Unit'}, inplace = True)
    convoutputpow = convoutputpow[convoutputpow.Unit.isin(convlist)]
    convoutputpow = convoutputpow.T
    convoutputpow.columns = convoutputpow.iloc[0]
    convoutputpow = convoutputpow[1:]
    scenarios[df_new_plots_name][f'TCONV_{s}'] = convoutputpow.sum(axis=1)
    
    # PARA ENCONTRAR TOTAL GENERATION (TG)
    scenarios[df_new_plots_name][f'TG_{s}'] = scenarios[results_name]['OutputPower'].sum(axis=1)
    
    # PARA ENCONTRAR PENETRATIO LEVEL (PL)
    scenarios[df_new_plots_name][f'PL_{s}'] = scenarios[df_new_plots_name][f'TRG_{s}']/scenarios[df_new_plots_name][f'TG_{s}']
    
    # PARA ENCONTRAR TOTAL CURTAILMENT (CURTAILMENT)
    scenarios[df_new_plots_name][f'Curtailment_{s}'] = scenarios[results_name]['OutputCurtailedPower'].sum(axis=1)/((scenarios[df_new_plots_name][f'TRG_{s}'])+scenarios[results_name]['OutputCurtailedPower'].sum(axis=1))
    
    # PARA ENCONTRAR TOTAL SHEDLOAD (SHEDLOAD)
    scenarios[df_new_plots_name][f'ShedLoad_{s}'] = scenarios[results_name]['OutputShedLoad'].sum(axis=1)
    
    # PARA ENCONTRAR INERTIA (INERTIA)
    OutputSysInertia = scenarios[results_name]['OutputSysInertia'].to_frame()
    OutputSysInertia.index = scenarios[df_new_plots_name].index
    scenarios[df_new_plots_name][f'SystemInertia_{s}'] = OutputSysInertia[0]
        
    # PARA ENCONTRAR GAIN (GAIN)
    OutputSystemGain = scenarios[results_name]['OutputSystemGain']
    if OutputSystemGain.empty:
        OutputSystemGain = pd.DataFrame(0, index=scenarios[df_new_plots_name].index, columns=[0])  # Cambia 'Column1' por el nombre de la columna deseada
        scenarios[df_new_plots_name][f'SystemGain_{s}'] = OutputSystemGain[0]
    else:    
        OutputSystemGain.index = scenarios[df_new_plots_name].index
        scenarios[df_new_plots_name][f'SystemGain_{s}'] = OutputSystemGain[0]
        
    # # PARA ENCONTRAR PrimaryReserve_Available (PrimaryReserve_Available)
    # OutputPrimaryReserve_Available = scenarios[results_name]['OutputPrimaryReserve_Available']
    # OutputPrimaryReserve_Available.index = scenarios[df_new_plots_name].index
    # scenarios[df_new_plots_name][f'PrimaryReserve_Available_{s}'] = OutputPrimaryReserve_Available
    
    # PARA ENCONTRAR PowerLoss (PowerLoss)
    OutputPowerLoss = scenarios[results_name]['OutputPowerLoss'].to_frame()
    OutputPowerLoss.index = scenarios[df_new_plots_name].index
    scenarios[df_new_plots_name][f'PowerLoss_{s}'] = OutputPowerLoss[0]
    
    # PARA ENCONTRAR EL HLimit SE DEBE CORRER LA FUNCION get_frequency_security_constraints
    # scenarios[contingency_name] = scenarios[contingency_name].set_index(scenarios[df_new_plots_name].index)
    scenarios[df_new_plots_name][f'InertiaLimit_{s}'] = scenarios[inputs_name]['param_df']['InertiaLimit']
    scenarios[df_new_plots_name][f'GainLimit_{s}'] = scenarios[inputs_name]['param_df']['SystemGainLimit']
    scenarios[df_new_plots_name][f'PrimaryReserveLimit_{s}'] = scenarios[inputs_name]['param_df']['PrimaryReserveLimit']


# #1. System Inertia

#     plt.style.use('default')
#     print(plt.style.available)    
    
#     df = scenarios[df_new_plots_name]
#     df[f'InertiaLimit_{s}']= scenarios[inputs_name]['param_df']['InertiaLimit']
    
#     # Especifica el intervalo de tiempo que deseas plotear
#     fecha_inicio = '2030-01-01'  # Reemplaza 'yyyy-mm-dd' con la fecha de inicio deseada
#     fecha_fin = '2030-12-31'     # Reemplaza 'yyyy-mm-dd' con la fecha de fin deseada
    
#     # Filtra las filas de OutputPower, PowerCapacity y OutputReserve_2U en el intervalo de tiempo especificado
#     df = df.loc[fecha_inicio:fecha_fin]
    
#     df['Year']=df.index.year
#     df['Month']=df.index.month
#     df['Day']=df.index.day
#     df['Hour']=df.index.hour
#     df['Week']=df.index.week
    
#     df = df.reset_index()
#     df.rename(columns={'index': 'TIMESTAMP'}, inplace=True)
#     df.set_index('TIMESTAMP', inplace=True)
    
#     plt.figure(figsize=(15,7))
#     sns.lineplot(data = df, y=f'SystemInertia_{s}', x = 'TIMESTAMP' , linewidth=1, color='teal',alpha=1,label=f'System_Inertia_{s}',n_boot=100)
#     sns.lineplot(data = df, y=f'InertiaLimit_{s}', x = 'TIMESTAMP', linewidth=1, color='red',alpha=0.5,label=f'Inertia_Limit_{s}',n_boot=100)      
#     plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize = 12)
#     plt.xticks(fontsize = 10)
#     plt.yticks(fontsize = 10)
#     plt.xlabel('TIMESTAMP', fontsize = 12)
#     plt.ylabel('System_Inertia[GW*s]', fontsize = 12)
#     plt.show()

# #2. Penetration level 


#     plt.style.use('default')
#     print(plt.style.available)
    
#     fig, ax = plt.subplots(figsize=(15,7))
#     sns.lineplot(data = df, y=f'PL_{s}', x = 'Week' , linewidth=1, color='teal',alpha=1,label=f'Penetration_Level_{s}')
#     plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize = 22)
#     plt.xticks(fontsize = 20)
#     plt.yticks(fontsize = 20)
#     plt.xlabel('Week', fontsize = 22)
#     plt.ylabel('Penetration_Level[pu]', fontsize = 22)
#     plt.show()

# #3. Curtailment 

#     plt.style.use('default')
#     print(plt.style.available)
    
#     fig, ax = plt.subplots(figsize=(15,7))
#     sns.lineplot(data = df, y=f'Curtailment_{s}', x = 'Week' , linewidth=1, color='teal',alpha=1,label=f'Curtailment_{s}')
#     plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize = 22)
#     plt.xticks(fontsize = 20)
#     plt.yticks(fontsize = 20)
#     plt.xlabel('Week', fontsize = 22)
#     plt.ylabel('Curtailment[pu]', fontsize = 22)
#     plt.show()
#%%
# PREPROCESSING


# sce_list = ['S0','S1','S2','S3','S4','S5','S6']
# sce_list = ['S0_SI','S1_SI','S2_SI','S3_SI','S4_SI','S5_SI','S6_SI']
# sce_list = ['S0_DI','S1_DI','S2_DI','S3_DI','S4_DI','S5_DI','S6_DI']
sce_list = [
            'S0','S0_DI','S0_SI',
            'S1','S1_DI','S1_SI',
            'S2','S2_DI','S2_SI',
            'S3','S3_DI','S3_SI',
            'S4','S4_DI','S4_SI',
            'S5','S5_DI','S5_SI'
            ]


for s in sce_list:
    scenarios[f'df_new_plots_{s}'] = scenarios[f'df_new_plots_{s}'].reset_index()
    scenarios[f'df_new_plots_{s}'].rename(columns={'index': 'TIMESTAMP'}, inplace=True)


for s in sce_list:
    scenarios[f'df_new_plots_{s}']['Year'] = scenarios[f'df_new_plots_{s}']['TIMESTAMP'].dt.year
    scenarios[f'df_new_plots_{s}']['Month'] = scenarios[f'df_new_plots_{s}']['TIMESTAMP'].dt.month
    scenarios[f'df_new_plots_{s}']['Day'] = scenarios[f'df_new_plots_{s}']['TIMESTAMP'].dt.day
    scenarios[f'df_new_plots_{s}']['Hour'] = scenarios[f'df_new_plots_{s}']['TIMESTAMP'].dt.hour
    scenarios[f'df_new_plots_{s}']['Week'] = scenarios[f'df_new_plots_{s}']['TIMESTAMP'].dt.week





#%% GRID DE INERCIAS HORARIAS
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8-whitegrid')  # Estilo profesional
%matplotlib inline

# Definir una paleta de colores más profesional
palette = sns.color_palette("cubehelix", 6)

# Crear la figura y los ejes en formato de grid 2x3
fig, axs = plt.subplots(2, 3, figsize=(20, 10))

# Lista de escenarios
scenarios_list = ['0', '1', '2', '3', '4', '5']

# Iterar sobre los escenarios y sobre los ejes correspondientes
for idx, scenario in enumerate(scenarios_list):
    ax = axs[idx // 3, idx % 3]  # Seleccionar el eje correspondiente (2x3 grid)
    
    # Graficar usando la nueva paleta de colores y ajustando la opacidad
    sns.lineplot(data=scenarios[f'df_new_plots_S{scenario}'], y=f'SystemInertia_S{scenario}', x='TIMESTAMP', linewidth=2.5, color=palette[0], label=f'System Inertia BL{scenario}', ax=ax)
    sns.lineplot(data=scenarios[f'df_new_plots_S{scenario}'], y=f'InertiaLimit_S{scenario}', x='TIMESTAMP', linewidth=2.5, color=palette[1], linestyle='--', label=f'Inertia Limit {scenario}', ax=ax)

    sns.lineplot(data=scenarios[f'df_new_plots_S{scenario}_SI'], y=f'SystemInertia_S{scenario}_SI', x='TIMESTAMP', linewidth=2, color=palette[2], label=f'System Inertia SC{scenario}', ax=ax)
    sns.lineplot(data=scenarios[f'df_new_plots_S{scenario}_SI'], y=f'InertiaLimit_S{scenario}_SI', x='TIMESTAMP', linewidth=2, color=palette[3], linestyle='--', label=f'Inertia Limit {scenario} SI', ax=ax)

    sns.lineplot(data=scenarios[f'df_new_plots_S{scenario}_DI'], y=f'SystemInertia_S{scenario}_DI', x='TIMESTAMP', linewidth=2, color=palette[4], label=f'System Inertia DC{scenario}', ax=ax)
    sns.lineplot(data=scenarios[f'df_new_plots_S{scenario}_DI'], y=f'InertiaLimit_S{scenario}_DI', x='TIMESTAMP', linewidth=2, color=palette[5], linestyle='--', label=f'Inertia Limit {scenario} DI', ax=ax)

    # Títulos y etiquetas por subplot
    ax.set_title(f'Scenario {scenario}', fontsize=14)
    ax.set_xlabel('Timestamp', fontsize=12)
    ax.set_ylabel('System Inertia [GW*s]', fontsize=12)
    ax.tick_params(axis='x', rotation=30)  # Rotar etiquetas de eje x
    ax.legend(fontsize=10)

# Ajustar la disposición para que los subplots no se solapen
plt.tight_layout()
plt.show()

#%% GRID DE INERCIAS SEMANALES 
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
plt.style.use('default')  # Usar estilo por defecto
%matplotlib inline

# Crear la figura y los ejes en formato de grid 3x2
fig, axs = plt.subplots(3, 2, figsize=(20, 15))  # Ajuste de filas y columnas

# Lista de escenarios
scenarios_list = ['0', '1', '2', '3', '4', '5']

# Definir una paleta de colores más profesional
palette = sns.color_palette("cubehelix", 6)

# Calcular los límites del eje y (mínimo y máximo) para mantener la escala igual en todos los subplots
inertia_min = float('inf')
inertia_max = float('-inf')

for scenario in scenarios_list:
    # Encontrar los valores mínimos y máximos de SystemInertia y InertiaLimit en todos los escenarios
    inertia_min = min(inertia_min,
                      scenarios[f'df_new_plots_S{scenario}'][f'SystemInertia_S{scenario}'].min(),
                      scenarios[f'df_new_plots_S{scenario}'][f'InertiaLimit_S{scenario}'].min(),
                      scenarios[f'df_new_plots_S{scenario}_SI'][f'SystemInertia_S{scenario}_SI'].min(),
                      scenarios[f'df_new_plots_S{scenario}_SI'][f'InertiaLimit_S{scenario}_SI'].min(),
                      scenarios[f'df_new_plots_S{scenario}_DI'][f'SystemInertia_S{scenario}_DI'].min(),
                      scenarios[f'df_new_plots_S{scenario}_DI'][f'InertiaLimit_S{scenario}_DI'].min())
    
    inertia_max = max(inertia_max,
                      scenarios[f'df_new_plots_S{scenario}'][f'SystemInertia_S{scenario}'].max(),
                      scenarios[f'df_new_plots_S{scenario}'][f'InertiaLimit_S{scenario}'].max(),
                      scenarios[f'df_new_plots_S{scenario}_SI'][f'SystemInertia_S{scenario}_SI'].max(),
                      scenarios[f'df_new_plots_S{scenario}_SI'][f'InertiaLimit_S{scenario}_SI'].max(),
                      scenarios[f'df_new_plots_S{scenario}_DI'][f'SystemInertia_S{scenario}_DI'].max(),
                      scenarios[f'df_new_plots_S{scenario}_DI'][f'InertiaLimit_S{scenario}_DI'].max())

# Iterar sobre los escenarios y sobre los ejes correspondientes
for idx, scenario in enumerate(scenarios_list):
    ax = axs[idx // 2, idx % 2]  # Seleccionar el eje correspondiente (3x2 grid)
    
    # Graficar usando la nueva paleta de colores y ajustando la opacidad
    sns.lineplot(data=scenarios[f'df_new_plots_S{scenario}'], y=f'SystemInertia_S{scenario}', x='Week', linewidth=2, color=palette[0], label=f'System Inertia BL{scenario}', ax=ax)
    sns.lineplot(data=scenarios[f'df_new_plots_S{scenario}'], y=f'InertiaLimit_S{scenario}', x='Week', linewidth=2, color=palette[1], linestyle='-.', label='Inertia Limit BL', ax=ax)

    sns.lineplot(data=scenarios[f'df_new_plots_S{scenario}_SI'], y=f'SystemInertia_S{scenario}_SI', x='Week', linewidth=2, color=palette[2], label=f'System Inertia SC{scenario}', ax=ax)
    sns.lineplot(data=scenarios[f'df_new_plots_S{scenario}_SI'], y=f'InertiaLimit_S{scenario}_SI', x='Week', linewidth=2, color=palette[3], linestyle='--', label='Inertia Limit SC', ax=ax)

    sns.lineplot(data=scenarios[f'df_new_plots_S{scenario}_DI'], y=f'SystemInertia_S{scenario}_DI', x='Week', linewidth=2, color=palette[4], label=f'System Inertia DC{scenario}', ax=ax)
    sns.lineplot(data=scenarios[f'df_new_plots_S{scenario}_DI'], y=f'InertiaLimit_S{scenario}_DI', x='Week', linewidth=2, color=palette[5], linestyle=':', label='Inertia Limit DC', ax=ax)

    # Establecer los mismos límites en el eje y para todos los subplots
    ax.set_ylim(-2, 30)

    # Títulos y etiquetas por subplot
    ax.set_title(f'Scenario {scenario}', fontsize=20)  # Aumentar tamaño de título (de 18 a 20)
    ax.set_xlabel('Week', fontsize=18)  # Aumentar tamaño de etiqueta en x (de 16 a 18)
    ax.set_ylabel('System Inertia [GW*s]', fontsize=18)  # Aumentar tamaño de etiqueta en y (de 16 a 18)
    ax.tick_params(axis='x', rotation=0, labelsize=16)  # Aumentar tamaño de ticks en x (de 14 a 16)
    ax.tick_params(axis='y', labelsize=16)  # Aumentar tamaño de ticks en y (de 14 a 16)
    ax.legend(fontsize=16)  # Aumentar tamaño de la leyenda (de 14 a 16)

# Ajustar la disposición para que los subplots no se solapen
plt.tight_layout()
plt.show()



#%% GRID DE FRECUENCIA DE OCURRENCIA DE INERCIAS 
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
plt.style.use('default')  # Usar estilo por defecto
%matplotlib inline

# Crear la figura y los ejes en formato de grid 3x2 (3 filas, 2 columnas)
fig, axs = plt.subplots(3, 2, figsize=(20, 15))  # Ajustado para 3 filas y 2 columnas

# Lista de escenarios
scenarios_list = ['0', '1', '2', '3', '4', '5']
palette = sns.color_palette("cubehelix", 6)  # Paleta de colores más bonita

# Definir el rango para el eje y (ajusta según tus datos)
y_lim = (0, 8000)  # Cambia este rango según el máximo esperado de frecuencia en tus datos

# Iterar sobre los escenarios y sobre los ejes correspondientes
for idx, scenario in enumerate(scenarios_list):
    ax = axs[idx // 2, idx % 2]  # Seleccionar el eje correspondiente (3x2 grid)
    
    # Graficar histogramas de System Inertia
    sns.histplot(data=scenarios[f'df_new_plots_S{scenario}'], 
                 x=f'SystemInertia_S{scenario}', 
                 bins=30, 
                 color=palette[0], 
                 label=f'System Inertia BL{scenario}', 
                 ax=ax, kde=False, 
                 stat='count', 
                 alpha=0.6)
    
    # Graficar histogramas de System Inertia SI
    sns.histplot(data=scenarios[f'df_new_plots_S{scenario}_SI'], 
                 x=f'SystemInertia_S{scenario}_SI', 
                 bins=30, 
                 color=palette[2], 
                 label=f'System Inertia SC{scenario}', 
                 ax=ax, kde=False, 
                 stat='count', 
                 alpha=0.6)
    
    # Graficar histogramas de System Inertia DI
    sns.histplot(data=scenarios[f'df_new_plots_S{scenario}_DI'], 
                 x=f'SystemInertia_S{scenario}_DI', 
                 bins=30, 
                 color=palette[4], 
                 label=f'System Inertia DC{scenario}', 
                 ax=ax, kde=False, 
                 stat='count', 
                 alpha=0.6)

    # Establecer los límites del eje y
    ax.set_ylim(y_lim)

    # Calcular estadísticas para cada System Inertia
    stats = {}
    for inertia_type in ['', '_SI', '_DI']:
        key = f'SystemInertia_S{scenario}{inertia_type}'
        stats[inertia_type] = {
            'mean': scenarios[f'df_new_plots_S{scenario}{inertia_type}'][key].mean(),
            'std': scenarios[f'df_new_plots_S{scenario}{inertia_type}'][key].std(),
            'min': scenarios[f'df_new_plots_S{scenario}{inertia_type}'][key].min(),
            'max': scenarios[f'df_new_plots_S{scenario}{inertia_type}'][key].max(),
        }

    # Añadir las estadísticas al gráfico
    box_props = dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.5')  # Propiedades del cuadro

    # Ajustar la posición de las estadísticas para aumentar la separación
    for i, (inertia_label, stat) in enumerate(stats.items()):
        ax.text(0.05, 0.95 - i * 0.32,  # Cambiado de 0.25 a 0.35 para aumentar la separación
                f'Mean {inertia_label}: {stat["mean"]:.2f}\nStd: {stat["std"]:.2f}\nMin: {stat["min"]:.2f}\nMax: {stat["max"]:.2f}', 
                transform=ax.transAxes, fontsize=16, verticalalignment='top', bbox=box_props, color=palette[i*2])  # Aumentado a 16

    # Mover la leyenda a la esquina superior derecha
    ax.legend(loc='upper right', fontsize=16)  # Aumentado a 16

    # Títulos y etiquetas por subplot
    ax.set_title(f'Scenario {scenario}', fontsize=20)  # Aumentado a 20
    ax.set_xlabel('System Inertia [GW*s]', fontsize=18)  # Aumentado a 18
    ax.set_ylabel('Frequency of Occurrence', fontsize=18)  # Aumentado a 18

    # Ajustar el tamaño de fuente de los ticks
    ax.tick_params(axis='both', which='major', labelsize=16)  # Tamaño de ticks en X e Y

# Ajustar la disposición para que los subplots no se solapen
plt.tight_layout()
plt.show()


#%% GRID DE PENETRACION SEMANALES
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
plt.style.use('default')  # Usar estilo por defecto
%matplotlib inline

# Crear la figura y los ejes en formato de grid 2x3 (2 columnas y 3 filas)
fig, axs = plt.subplots(3, 2, figsize=(20, 10))  # Ajustar el tamaño según la nueva disposición

# Lista de escenarios
scenarios_list = ['0', '1', '2', '3', '4', '5']

# Definir una paleta de colores más profesional
palette = sns.color_palette("cubehelix", 6)

# Iterar sobre los escenarios y sobre los ejes correspondientes
for idx, scenario in enumerate(scenarios_list):
    ax = axs[idx // 2, idx % 2]  # Seleccionar el eje correspondiente (3 filas, 2 columnas)
    
    # Graficar usando la nueva paleta de colores y ajustando la opacidad
    sns.lineplot(data=scenarios[f'df_new_plots_S{scenario}'], 
                 y=scenarios[f'df_new_plots_S{scenario}'][f'PL_S{scenario}'] * 100,  # Multiplicar por 100
                 x='Week', 
                 linewidth=1.5, 
                 color=palette[0], 
                 label=f'VRE Penetration BL{scenario}', 
                 ax=ax)

    sns.lineplot(data=scenarios[f'df_new_plots_S{scenario}_SI'], 
                 y=scenarios[f'df_new_plots_S{scenario}_SI'][f'PL_S{scenario}_SI'] * 100,  # Multiplicar por 100
                 x='Week', 
                 linewidth=1.5, 
                 color=palette[2], 
                 label=f'VRE Penetration SC{scenario}', 
                 ax=ax)

    sns.lineplot(data=scenarios[f'df_new_plots_S{scenario}_DI'], 
                 y=scenarios[f'df_new_plots_S{scenario}_DI'][f'PL_S{scenario}_DI'] * 100,  # Multiplicar por 100
                 x='Week', 
                 linewidth=1.5, 
                 color=palette[4], 
                 label=f'VRE Penetration DC{scenario}', 
                 ax=ax)

    # Establecer los límites en el eje y y los ticks en porcentaje
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100])  # Establecer los ticks del eje y en porcentaje
    ax.set_yticklabels([f'{tick}' for tick in ax.get_yticks()])  # Formatear ticks como %

    # Títulos y etiquetas por subplot
    ax.set_title(f'Scenario {scenario}', fontsize=20)  # Aumentado a 20
    ax.set_xlabel('Week', fontsize=18)  # Aumentado a 18
    ax.set_ylabel('VRE Penetration [%]', fontsize=18)  # Cambiar etiqueta a %
    ax.tick_params(axis='x', rotation=0)  # No rotar etiquetas de eje x
    
    # Leyenda en la esquina superior izquierda
    ax.legend(fontsize=16, loc='upper left')  # Aumentado a 16 y ubicado en la esquina superior izquierda

    # Establecer tamaño de fuente de los ticks
    ax.tick_params(axis='both', which='major', labelsize=16)  # Tamaño de ticks en X e Y

# Ajustar la disposición para que los subplots no se solapen
plt.tight_layout(pad=0.5)  # Aumentar el padding entre los subplots
plt.show()




#%% GRID DE CURTAILMENT SEMANALES
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
plt.style.use('default')  # Usar estilo por defecto
%matplotlib inline

# Crear la figura y los ejes en formato de grid 3x2
fig, axs = plt.subplots(3, 2, figsize=(20, 10))  # Ajustar la altura de la figura

# Lista de escenarios
scenarios_list = ['0', '1', '2', '3', '4', '5']

# Definir una paleta de colores más profesional
palette = sns.color_palette("cubehelix", 6)

# Calcular los límites del eje y (mínimo y máximo) para mantener la escala igual en todos los subplots
curtailment_min = float('inf')
curtailment_max = float('-inf')

for scenario in scenarios_list:
    # Encontrar los valores mínimos y máximos de Curtailment en todos los escenarios
    curtailment_min = min(curtailment_min,
                          scenarios[f'df_new_plots_S{scenario}'][f'Curtailment_S{scenario}'].min(),
                          scenarios[f'df_new_plots_S{scenario}_SI'][f'Curtailment_S{scenario}_SI'].min(),
                          scenarios[f'df_new_plots_S{scenario}_DI'][f'Curtailment_S{scenario}_DI'].min())
    
    curtailment_max = max(curtailment_max,
                          scenarios[f'df_new_plots_S{scenario}'][f'Curtailment_S{scenario}'].max(),
                          scenarios[f'df_new_plots_S{scenario}_SI'][f'Curtailment_S{scenario}_SI'].max(),
                          scenarios[f'df_new_plots_S{scenario}_DI'][f'Curtailment_S{scenario}_DI'].max())

# Iterar sobre los escenarios y sobre los ejes correspondientes
for idx, scenario in enumerate(scenarios_list):
    ax = axs[idx // 2, idx % 2]  # Seleccionar el eje correspondiente (3 filas, 2 columnas)
    
    # Graficar usando la nueva paleta de colores y ajustando la opacidad (multiplicando los valores por 100)
    sns.lineplot(data=scenarios[f'df_new_plots_S{scenario}'], 
                 y=scenarios[f'df_new_plots_S{scenario}'][f'Curtailment_S{scenario}'] * 100, 
                 x='Week', 
                 linewidth=1.5, 
                 color=palette[0], 
                 label=f'VRE Curtailment BL{scenario}', 
                 ax=ax)

    sns.lineplot(data=scenarios[f'df_new_plots_S{scenario}_SI'], 
                 y=scenarios[f'df_new_plots_S{scenario}_SI'][f'Curtailment_S{scenario}_SI'] * 100, 
                 x='Week', 
                 linewidth=1.5, 
                 color=palette[2], 
                 label=f'VRE Curtailment SC{scenario}', 
                 ax=ax)

    sns.lineplot(data=scenarios[f'df_new_plots_S{scenario}_DI'], 
                 y=scenarios[f'df_new_plots_S{scenario}_DI'][f'Curtailment_S{scenario}_DI'] * 100, 
                 x='Week', 
                 linewidth=1.5, 
                 color=palette[4], 
                 label=f'VRE Curtailment DC{scenario}', 
                 ax=ax)

    # Establecer los mismos límites en el eje y para todos los subplots
    ax.set_ylim(0, curtailment_max * 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100])  # Establecer los ticks del eje y en porcentaje

    # Títulos y etiquetas por subplot
    ax.set_title(f'Scenario {scenario}', fontsize=20)  # Aumentado a 20
    ax.set_xlabel('Week', fontsize=18, labelpad=10)  # Aumentado a 18 y se añade espacio extra
    ax.set_ylabel('VRE Curtailment [%]', fontsize=18, labelpad=10)  # Cambiado a porcentaje y aumentado a 18
    ax.tick_params(axis='x', rotation=0)  # No rotar etiquetas de eje x
    
    # Leyenda en la esquina superior derecha
    ax.legend(fontsize=16, loc='upper right')  # Aumentado a 16 y ubicado en la esquina superior derecha

    # Establecer tamaño de fuente de los ticks
    ax.tick_params(axis='both', which='major', labelsize=16)  # Tamaño de ticks en X e Y

# Ajustar la disposición para que los subplots no se solapen
plt.tight_layout(pad=0.5)  # Aumentar el padding entre los subplots
plt.show()



#%% 5. Cumulative distributions System Inertia in boxplots
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


df = pd.DataFrame()

# Lista de escenarios
scenarios_list = ['0', '1', '2', '3', '4', '5']

# Agregar columnas de System Inertia y HLimit al DataFrame
for scenario in scenarios_list:
    df[f'SystemInertia_S{scenario}'] = scenarios[f'df_new_plots_S{scenario}'][f'SystemInertia_S{scenario}']
    df[f'SystemInertia_S{scenario}_SI'] = scenarios[f'df_new_plots_S{scenario}_SI'][f'SystemInertia_S{scenario}_SI']
    df[f'SystemInertia_S{scenario}_DI'] = scenarios[f'df_new_plots_S{scenario}_DI'][f'SystemInertia_S{scenario}_DI']

# Ahora `df` tiene 18 columnas (6 escenarios * 3 columnas por escenario)
sns.set(style='whitegrid')

fig, ax = plt.subplots(figsize=(25, 9))
xvalues = [
    'BL0', 'SC0', 'DC0',
    'BL1', 'SC1', 'DC1',
    'BL2', 'SC2', 'DC2',
    'BL3', 'SC3', 'DC3',
    'BL4', 'SC4', 'DC4',
    'BL5', 'SC5', 'DC5'
]

palette = ['teal','teal','teal','darkslategrey','darkslategrey','darkslategrey','navy','navy','navy', 
           'seagreen', 'seagreen', 'seagreen', 'darkkhaki', 'darkkhaki', 'darkkhaki','plum','plum','plum']

# Asegurarse de que la longitud de `xvalues` coincida con las columnas en `df`
df.columns = xvalues

# Convertir el DataFrame a formato long para usar en el boxplot
melted_df = df.melt(var_name='Scenario', value_name='System Inertia')

# Dibujar el boxplot
sns.boxplot(data=melted_df, x='Scenario', y='System Inertia', palette=palette, ax=ax, width=0.7)

# Rotar y ajustar los xticks para evitar superposición
plt.xticks(rotation=0, ha='center', fontsize=20)
plt.yticks(fontsize=20)

# Cambiar el texto del eje x a "Scenarios"
ax.set_xlabel('Scenarios', fontsize=20)

# Cambiar el texto del eje y a "System Inertia [GW*s]"
ax.set_ylabel("System Inertia [GW*s]", fontsize=20)

# Agregar estadísticas clave (mediana, Q1, Q3, min, max) sobre los boxplots
stats_summary = df.describe().T[['min', '25%', '50%', '75%', 'max']]

# Mostrar la tabla de estadísticas
print(stats_summary)

# Análisis de las estadísticas clave:
# - Revisar diferencias entre los escenarios en cuanto a inercia mínima, máxima y distribución intercuartil.
# - Interpretar si algún escenario muestra menor variabilidad, lo cual podría ser una indicación de estabilidad.
# - Identificar escenarios con valores atípicos (outliers) y discutir su relevancia.

#%% 5.1 Cumulative distributions System Inertia in boxplots
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Crear la figura y los ejes en formato de grid 2x3
fig, axs = plt.subplots(2, 3, figsize=(25, 12))

# Lista de escenarios
scenarios_list = ['0', '1', '2', '3', '4', '5']

# Lista de colores para los diferentes escenarios
palette = ['teal', 'teal', 'teal', 'darkslategrey', 'darkslategrey', 'darkslategrey',
           'navy', 'navy', 'navy', 'seagreen', 'seagreen', 'seagreen', 
           'darkkhaki', 'darkkhaki', 'darkkhaki', 'plum', 'plum', 'plum']

# Definir los límites del eje y para todos los subplots
ymin, ymax = 0, 30  # Ajusta estos valores según tus datos

# Iterar sobre los escenarios y sobre los ejes correspondientes
for idx, scenario in enumerate(scenarios_list):
    ax = axs[idx // 3, idx % 3]  # Seleccionar el eje correspondiente (2x3 grid)
    
    # Obtener los datos del escenario actual
    df = pd.DataFrame()
    df[f'SystemInertia_S{scenario}'] = scenarios[f'df_new_plots_S{scenario}'][f'SystemInertia_S{scenario}']
    df[f'SystemInertia_S{scenario}_SI'] = scenarios[f'df_new_plots_S{scenario}_SI'][f'SystemInertia_S{scenario}_SI']
    df[f'SystemInertia_S{scenario}_DI'] = scenarios[f'df_new_plots_S{scenario}_DI'][f'SystemInertia_S{scenario}_DI']
    
    # Renombrar las columnas a algo más legible para el boxplot
    df.columns = [f'BL{scenario}', f'SC{scenario}', f'DC{scenario}']
    
    # Convertir a formato largo (long format) para el boxplot
    melted_df = df.melt(var_name='Scenario', value_name='System Inertia')
    
    # Graficar el boxplot para este escenario
    sns.boxplot(data=melted_df, x='Scenario', y='System Inertia', 
                palette=palette[idx*3:idx*3+3], ax=ax, width=0.7)
    
    # Añadir títulos y etiquetas
    ax.set_title(f'Scenario {scenario}', fontsize=14)
    ax.set_xlabel('', fontsize=12)
    ax.set_ylabel('System Inertia [GW*s]', fontsize=12)
    
    # Configurar el mismo límite en el eje y para todos los subplots
    ax.set_ylim(ymin, ymax)

    # Calcular estadísticas descriptivas
    stats = df.describe().T[['min', '25%', '50%', '75%', 'max']]  # Obtener estadísticas

    # Añadir estadísticas debajo de cada boxplot, alineadas y con el mismo tamaño que las etiquetas
    for i, column in enumerate(df.columns):
        min_val = stats.loc[column, 'min']
        q1 = stats.loc[column, '25%']
        median = stats.loc[column, '50%']
        q3 = stats.loc[column, '75%']
        max_val = stats.loc[column, 'max']
        
        # Colocar el texto debajo del gráfico
        ax.text(i, ymin - 2, f'Min: {min_val:.2f}\nQ1: {q1:.2f}\nMed: {median:.2f}\nQ3: {q3:.2f}\nMax: {max_val:.2f}',
                ha='center', va='top', fontsize=12, color='black')

    # Ajustar ticks
    ax.tick_params(axis='x', rotation=0, labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

# Ajustar la disposición para que los subplots no se solapen
plt.tight_layout()
plt.show()
#%% 5.2 Cumulative distributions System Inertia in boxplots
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Crear la figura y los ejes en formato de una fila con 6 subplots
fig, axs = plt.subplots(1, 6, figsize=(30, 8))  # Cambiar a 1 fila y 6 columnas

# Lista de escenarios
scenarios_list = ['0', '1', '2', '3', '4', '5']

# Lista de colores para los diferentes escenarios
palette = ['teal', 'teal', 'teal', 'darkslategrey', 'darkslategrey', 'darkslategrey',
           'navy', 'navy', 'navy', 'seagreen', 'seagreen', 'seagreen', 
           'darkkhaki', 'darkkhaki', 'darkkhaki', 'plum', 'plum', 'plum']

# Definir los límites del eje y para todos los subplots
ymin, ymax = 0, 30  # Ajusta estos valores según tus datos

# Iterar sobre los escenarios y sobre los ejes correspondientes
for idx, scenario in enumerate(scenarios_list):
    ax = axs[idx]  # Seleccionar el eje correspondiente (1 fila, 6 columnas)
    
    # Obtener los datos del escenario actual
    df = pd.DataFrame()
    df[f'SystemInertia_S{scenario}'] = scenarios[f'df_new_plots_S{scenario}'][f'SystemInertia_S{scenario}']
    df[f'SystemInertia_S{scenario}_SI'] = scenarios[f'df_new_plots_S{scenario}_SI'][f'SystemInertia_S{scenario}_SI']
    df[f'SystemInertia_S{scenario}_DI'] = scenarios[f'df_new_plots_S{scenario}_DI'][f'SystemInertia_S{scenario}_DI']
    
    # Renombrar las columnas a algo más legible para el boxplot
    df.columns = [f'BL{scenario}', f'SC{scenario}', f'DC{scenario}']
    
    # Convertir a formato largo (long format) para el boxplot
    melted_df = df.melt(var_name='Scenario', value_name='System Inertia')
    
    # Graficar el boxplot para este escenario
    sns.boxplot(data=melted_df, x='Scenario', y='System Inertia', 
                palette=palette[idx*3:idx*3+3], ax=ax, width=0.7)
    
    # Añadir títulos y etiquetas
    ax.set_title(f'Scenario {scenario}', fontsize=14)
    ax.set_xlabel('', fontsize=12)
    ax.set_ylabel('System Inertia [GW*s]', fontsize=12)
    
    # Configurar el mismo límite en el eje y para todos los subplots
    ax.set_ylim(ymin, ymax)

    # Calcular estadísticas descriptivas
    stats = df.describe().T[['min', '25%', '50%', '75%', 'max']]  # Obtener estadísticas

    # Añadir estadísticas debajo de cada boxplot, alineadas y con el mismo tamaño que las etiquetas
    for i, column in enumerate(df.columns):
        min_val = stats.loc[column, 'min']
        q1 = stats.loc[column, '25%']
        median = stats.loc[column, '50%']
        q3 = stats.loc[column, '75%']
        max_val = stats.loc[column, 'max']
        
        # Colocar el texto debajo del gráfico
        ax.text(i, ymin - 2, f'Min: {min_val:.2f}\nQ1: {q1:.2f}\nMed: {median:.2f}\nQ3: {q3:.2f}\nMax: {max_val:.2f}',
                ha='center', va='top', fontsize=12, color='black')

    # Ajustar ticks
    ax.tick_params(axis='x', rotation=0, labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

# Ajustar la disposición para que los subplots no se solapen
plt.tight_layout()
plt.show()
#%% 5.3 Cumulative distributions System Inertia in boxplots
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Crear la figura y los ejes en formato de una fila con 6 subplots, compartiendo el eje y
fig, axs = plt.subplots(1, 6, figsize=(24, 12), sharey=True)  # Aumentar el tamaño de la figura

# Lista de escenarios
scenarios_list = ['0', '1', '2', '3', '4', '5']

# Lista de colores para los diferentes escenarios
palette = ['teal', 'teal', 'teal', 'darkslategrey', 'darkslategrey', 'darkslategrey',
           'navy', 'navy', 'navy', 'seagreen', 'seagreen', 'seagreen', 
           'darkkhaki', 'darkkhaki', 'darkkhaki', 'plum', 'plum', 'plum']

# Definir los límites del eje y para todos los subplots
ymin, ymax = 0, 30  # Ajusta estos valores según tus datos

# Aumentar el tamaño de fuente de todo
font_size = 18  # Tamaño de las fuentes para todo el gráfico
stats_font_size = 16  # Tamaño de las fuentes de las estadísticas

# Iterar sobre los escenarios y sobre los ejes correspondientes
for idx, scenario in enumerate(scenarios_list):
    ax = axs[idx]  # Seleccionar el eje correspondiente (1 fila, 6 columnas)
    
    # Obtener los datos del escenario actual
    df = pd.DataFrame()
    df[f'SystemInertia_S{scenario}'] = scenarios[f'df_new_plots_S{scenario}'][f'SystemInertia_S{scenario}']
    df[f'SystemInertia_S{scenario}_SI'] = scenarios[f'df_new_plots_S{scenario}_SI'][f'SystemInertia_S{scenario}_SI']
    df[f'SystemInertia_S{scenario}_DI'] = scenarios[f'df_new_plots_S{scenario}_DI'][f'SystemInertia_S{scenario}_DI']
    
    # Renombrar las columnas a algo más legible para el boxplot
    df.columns = [f'BL{scenario}', f'SC{scenario}', f'DC{scenario}']
    
    # Convertir a formato largo (long format) para el boxplot
    melted_df = df.melt(var_name='Scenario', value_name='System Inertia')
    
    # Graficar el boxplot para este escenario
    sns.boxplot(data=melted_df, x='Scenario', y='System Inertia', 
                palette=palette[idx*3:idx*3+3], ax=ax, width=0.7)
    
    # Añadir títulos y etiquetas
    ax.set_title(f'Scenario {scenario}', fontsize=font_size + 6)  # Aumentar el tamaño del título
    ax.set_xlabel('', fontsize=font_size)  # No es necesario modificar los ejes x individualmente
    if idx == 0:
        ax.set_ylabel('System Inertia [GW*s]', fontsize=font_size + 4)  # Solo un eje y compartido
    else:
        ax.set_ylabel('')  # Quitar los ejes y de los subplots siguientes
    
    # Configurar el mismo límite en el eje y para todos los subplots
    ax.set_ylim(ymin, ymax + 5)  # Dejar espacio adicional para las estadísticas

    # Calcular estadísticas descriptivas
    stats = df.describe().T[['min', '25%', '50%', '75%', 'max']]  # Obtener estadísticas

    # Añadir estadísticas debajo de cada boxplot, alineadas y con el mismo tamaño que las etiquetas
    for i, column in enumerate(df.columns):
        min_val = stats.loc[column, 'min']
        q1 = stats.loc[column, '25%']
        median = stats.loc[column, '50%']
        q3 = stats.loc[column, '75%']
        max_val = stats.loc[column, 'max']
        
        # Colocar el texto debajo del gráfico con un mayor tamaño de fuente para las estadísticas
        ax.text(i, ymin - 8, f'Min: {min_val:.2f}\nQ1: {q1:.2f}\nMed: {median:.2f}\nQ3: {q3:.2f}\nMax: {max_val:.2f}',
                ha='center', va='top', fontsize=stats_font_size, color='black')  # Aumentar tamaño de las estadísticas

    # Ajustar ticks
    ax.tick_params(axis='x', rotation=0, labelsize=font_size)
    ax.tick_params(axis='y', labelsize=font_size)

# Ajustar la disposición para que los subplots no se solapen
plt.tight_layout()
plt.show()


#%% 5.4 Cumulative distributions System Inertia in boxplots
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Crear la figura y los ejes con un solo eje y compartido
fig, axs = plt.subplots(1, 6, figsize=(25, 6), sharey=True)  # Cambiar tamaño de figura para una hoja LaTeX

# Lista de escenarios
scenarios_list = ['0', '1', '2', '3', '4', '5']

# Lista de colores para los diferentes escenarios
palette = ['teal', 'teal', 'teal', 'darkslategrey', 'darkslategrey', 'darkslategrey',
           'navy', 'navy', 'navy', 'seagreen', 'seagreen', 'seagreen', 
           'darkkhaki', 'darkkhaki', 'darkkhaki', 'plum', 'plum', 'plum']

# Definir los límites del eje y para todos los subplots
ymin, ymax = 0, 30  # Ajusta estos valores según tus datos

# Aumentar el tamaño de fuente
fontsize_labels = 16
fontsize_ticks = 14
fontsize_title = 18
fontsize_stats = 12

# Iterar sobre los escenarios y sobre los ejes correspondientes
for idx, scenario in enumerate(scenarios_list):
    ax = axs[idx]  # Seleccionar el eje correspondiente (1 fila, 6 columnas)
    
    # Obtener los datos del escenario actual
    df = pd.DataFrame()
    df[f'SystemInertia_S{scenario}'] = scenarios[f'df_new_plots_S{scenario}'][f'SystemInertia_S{scenario}']
    df[f'SystemInertia_S{scenario}_SI'] = scenarios[f'df_new_plots_S{scenario}_SI'][f'SystemInertia_S{scenario}_SI']
    df[f'SystemInertia_S{scenario}_DI'] = scenarios[f'df_new_plots_S{scenario}_DI'][f'SystemInertia_S{scenario}_DI']
    
    # Renombrar las columnas a algo más legible para el boxplot
    df.columns = [f'BL{scenario}', f'SC{scenario}', f'DC{scenario}']
    
    # Convertir a formato largo (long format) para el boxplot
    melted_df = df.melt(var_name='Scenario', value_name='System Inertia')
    
    # Graficar el boxplot para este escenario
    sns.boxplot(data=melted_df, x='Scenario', y='System Inertia', 
                palette=palette[idx*3:idx*3+3], ax=ax, width=0.7)
    
    # Añadir título con mayor tamaño de fuente
    ax.set_title(f'Scenario {scenario}', fontsize=fontsize_title)
    
    # Deshabilitar la etiqueta del eje y para los subplots después del primero
    if idx == 0:
        ax.set_ylabel('System Inertia [GW*s]', fontsize=fontsize_labels)
    else:
        ax.set_ylabel('')
    
    # Deshabilitar las etiquetas del eje x para mantener solo los ticks
    ax.set_xlabel('', fontsize=fontsize_labels)
    
    # Configurar el mismo límite en el eje y para todos los subplots
    ax.set_ylim(ymin, ymax)

    # Calcular estadísticas descriptivas
    stats = df.describe().T[['min', '25%', '50%', '75%', 'max']]  # Obtener estadísticas

    # Añadir estadísticas debajo de cada boxplot, alineadas y con el mismo tamaño que las etiquetas
    for i, column in enumerate(df.columns):
        min_val = stats.loc[column, 'min']
        q1 = stats.loc[column, '25%']
        median = stats.loc[column, '50%']
        q3 = stats.loc[column, '75%']
        max_val = stats.loc[column, 'max']
        
        # Colocar el texto debajo del gráfico con mayor tamaño de fuente
        ax.text(i, ymin - 2, f'Min: {min_val:.2f}\nQ1: {q1:.2f}\nMed: {median:.2f}\nQ3: {q3:.2f}\nMax: {max_val:.2f}',
                ha='center', va='top', fontsize=fontsize_stats, color='black')

    # Ajustar los ticks para que el texto no se solape
    ax.tick_params(axis='x', rotation=0, labelsize=fontsize_ticks)
    ax.tick_params(axis='y', labelsize=fontsize_ticks)

# Ajustar la disposición para que los subplots no se solapen
plt.tight_layout()
plt.show()
#%% 5.5 Cumulative distributions System Inertia in boxplots
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Crear la figura y los ejes en formato de 1 fila y 6 columnas
fig, axs = plt.subplots(1, 6, figsize=(20, 10), sharey=True)  # Ajustar tamaño para 6x1

# Lista de escenarios
scenarios_list = ['0', '1', '2', '3', '4', '5']

# Lista de colores para los diferentes escenarios (usaremos índices [0], [2], [4])
palette = sns.color_palette("cubehelix", 6)

# Definir los límites del eje y para todos los subplots
ymin, ymax = 0, 30  # Ajusta estos valores según tus datos

# Aumentar el tamaño de fuente de todo
font_size = 20  # Tamaño de las fuentes para todo el gráfico
stats_font_size = 16  # Tamaño de las fuentes de las estadísticas

# Ajustar el grosor de los boxplots
boxplot_width = 0.5

# Iterar sobre los escenarios y sobre los ejes correspondientes
for idx, scenario in enumerate(scenarios_list):
    ax = axs[idx]  # Seleccionar el eje correspondiente
    
    # Obtener los datos del escenario actual
    df = pd.DataFrame()
    df[f'SystemInertia_S{scenario}'] = scenarios[f'df_new_plots_S{scenario}'][f'SystemInertia_S{scenario}']
    df[f'SystemInertia_S{scenario}_SI'] = scenarios[f'df_new_plots_S{scenario}_SI'][f'SystemInertia_S{scenario}_SI']
    df[f'SystemInertia_S{scenario}_DI'] = scenarios[f'df_new_plots_S{scenario}_DI'][f'SystemInertia_S{scenario}_DI']
    
    # Renombrar las columnas a algo más legible para el boxplot
    df.columns = [f'BL{scenario}', f'SC{scenario}', f'DC{scenario}']
    
    # Convertir a formato largo (long format) para el boxplot
    melted_df = df.melt(var_name='Scenario', value_name='System Inertia')
    
    # Graficar el boxplot para este escenario usando los colores en los índices [0], [2], [4]
    sns.boxplot(data=melted_df, x='Scenario', y='System Inertia', 
                palette=[palette[0], palette[2], palette[4]], ax=ax, width=boxplot_width)
    
    # Añadir títulos y etiquetas
    ax.set_title(f'Scenario {scenario}', fontsize=font_size + 4)  # Aumentar el tamaño del título
    ax.set_xlabel('', fontsize=font_size)
    if idx == 0:
        ax.set_ylabel('System Inertia [GW*s]', fontsize=font_size + 2)  # Solo un eje y compartido
    else:
        ax.set_ylabel('')  # Quitar los ejes y de los subplots siguientes
    
    # Configurar el mismo límite en el eje y para todos los subplots
    ax.set_ylim(ymin, ymax + 5)  # Dejar espacio adicional para las estadísticas

    # Calcular estadísticas descriptivas
    stats = df.describe().T[['min', '25%', '50%', '75%', 'max']]  # Obtener estadísticas

    # Añadir un cuadro con todas las estadísticas para cada boxplot
    min_vals = [f'{stats.loc[col, "min"]:.2f}' for col in df.columns]
    q1_vals = [f'{stats.loc[col, "25%"]:.2f}' for col in df.columns]
    med_vals = [f'{stats.loc[col, "50%"]:.2f}' for col in df.columns]
    q3_vals = [f'{stats.loc[col, "75%"]:.2f}' for col in df.columns]
    max_vals = [f'{stats.loc[col, "max"]:.2f}' for col in df.columns]

    # Colocar el texto debajo del gráfico con mayor tamaño de fuente
    stats_text = (
        f"Min:    {min_vals[0]}  {min_vals[1]}   {min_vals[2]}\n"
        f"Q1:    {q1_vals[0]}  {q1_vals[1]}  {q1_vals[2]}\n"
        f"Med:  {med_vals[0]}  {med_vals[1]}  {med_vals[2]}\n"
        f"Q3:    {q3_vals[0]}  {q3_vals[1]}  {q3_vals[2]}\n"
        f"Max:  {max_vals[0]}  {max_vals[1]}  {max_vals[2]}"
    )
    
    # Añadir el texto dentro del gráfico
    ax.text(0.5, ymin - 5, stats_text, ha='center', va='top', fontsize=stats_font_size,
            bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='lightgrey'))

    # Ajustar los ticks para que el texto no se solape
    ax.tick_params(axis='x', rotation=0, labelsize=font_size)
    ax.tick_params(axis='y', labelsize=font_size)

# Ajustar la disposición para que los subplots no se solapen
plt.tight_layout(pad=2.0)  # Aumentar el padding entre los subplots
plt.show()

#%% 6. Grid de PowerLoss frequency of occurrence
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
plt.style.use('default')  # Usar estilo por defecto
%matplotlib inline

# Crear la figura y los ejes en formato de grid 2x3
fig, axs = plt.subplots(2, 3, figsize=(20, 10))

# Lista de escenarios
scenarios_list = ['0', '1', '2', '3', '4', '5']
palette = sns.color_palette("cubehelix", 6)  # Paleta de colores

# Definir el rango para el eje y (ajusta según tus datos)
y_lim = (0, 4000)  # Cambia este rango según el máximo esperado de frecuencia en tus datos

# Iterar sobre los escenarios y sobre los ejes correspondientes
for idx, scenario in enumerate(scenarios_list):
    ax = axs[idx // 3, idx % 3]  # Seleccionar el eje correspondiente (2x3 grid)
    
    # Graficar histogramas de PowerLoss para cada escenario (BL, SI, DI) con estilo uniforme
    sns.histplot(data=scenarios[f'df_new_plots_S{scenario}'], x=f'PowerLoss_S{scenario}', bins=30, color=palette[0], label=f'PowerLoss BL{scenario}', ax=ax, kde=False, stat='count', alpha=0.6)
    
    sns.histplot(data=scenarios[f'df_new_plots_S{scenario}_SI'], x=f'PowerLoss_S{scenario}_SI', bins=30, color=palette[2], label=f'PowerLoss SC{scenario}', ax=ax, kde=False, stat='count', alpha=0.6)
    
    sns.histplot(data=scenarios[f'df_new_plots_S{scenario}_DI'], x=f'PowerLoss_S{scenario}_DI', bins=30, color=palette[4], label=f'PowerLoss DC{scenario}', ax=ax, kde=False, stat='count', alpha=0.6)

    # Establecer los límites del eje y
    ax.set_ylim(y_lim)

    # Calcular estadísticas (media y desviación estándar) para cada PowerLoss
    mean_pl = scenarios[f'df_new_plots_S{scenario}'][f'PowerLoss_S{scenario}'].mean()
    std_pl = scenarios[f'df_new_plots_S{scenario}'][f'PowerLoss_S{scenario}'].std()

    mean_pl_si = scenarios[f'df_new_plots_S{scenario}_SI'][f'PowerLoss_S{scenario}_SI'].mean()
    std_pl_si = scenarios[f'df_new_plots_S{scenario}_SI'][f'PowerLoss_S{scenario}_SI'].std()

    mean_pl_di = scenarios[f'df_new_plots_S{scenario}_DI'][f'PowerLoss_S{scenario}_DI'].mean()
    std_pl_di = scenarios[f'df_new_plots_S{scenario}_DI'][f'PowerLoss_S{scenario}_DI'].std()

    # Añadir las estadísticas al gráfico (esquina superior izquierda)
    ax.text(0.05, 0.95, f'Mean BL: {mean_pl:.2f}\nStd: {std_pl:.2f}', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8), color=palette[0])

    ax.text(0.05, 0.80, f'Mean SC: {mean_pl_si:.2f}\nStd SI: {std_pl_si:.2f}', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8), color=palette[2])

    ax.text(0.05, 0.65, f'Mean DC: {mean_pl_di:.2f}\nStd DI: {std_pl_di:.2f}', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8), color=palette[4])

    # Mover la leyenda a la esquina superior derecha
    ax.legend(loc='upper right', fontsize=10)

    # Títulos y etiquetas por subplot
    ax.set_title(f'Scenario {scenario}', fontsize=14)
    ax.set_xlabel('PowerLoss', fontsize=12)
    ax.set_ylabel('Frequency of Occurrence', fontsize=12)

# Ajustar la disposición para que los subplots no se solapen
plt.tight_layout()
plt.show()
#%% 7. Grid de PowerLoss density
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
plt.style.use('default')  # Usar estilo por defecto
%matplotlib inline

# Crear la figura y los ejes en formato de grid 2x3
fig, axs = plt.subplots(2, 3, figsize=(20, 10))

# Lista de escenarios
scenarios_list = ['0', '1', '2', '3', '4', '5']
palette = sns.color_palette("cubehelix", 6)  # Paleta de colores

# Iterar sobre los escenarios y sobre los ejes correspondientes
for idx, scenario in enumerate(scenarios_list):
    ax = axs[idx // 3, idx % 3]  # Seleccionar el eje correspondiente (2x3 grid)
    
    # Graficar kdeplot de PowerLoss para cada escenario (BL, SI, DI) con estilo uniforme
    sns.kdeplot(data=scenarios[f'df_new_plots_S{scenario}'], x=f'PowerLoss_S{scenario}', color=palette[0], label=f'PowerLoss BL{scenario}', ax=ax, fill=True, alpha=0.6)
    
    sns.kdeplot(data=scenarios[f'df_new_plots_S{scenario}_SI'], x=f'PowerLoss_S{scenario}_SI', color=palette[2], label=f'PowerLoss SC{scenario}', ax=ax, fill=True, alpha=0.6)
    
    sns.kdeplot(data=scenarios[f'df_new_plots_S{scenario}_DI'], x=f'PowerLoss_S{scenario}_DI', color=palette[4], label=f'PowerLoss DC{scenario}', ax=ax, fill=True, alpha=0.6)

    # Calcular estadísticas (media y desviación estándar) para cada PowerLoss
    mean_pl = scenarios[f'df_new_plots_S{scenario}'][f'PowerLoss_S{scenario}'].mean()
    std_pl = scenarios[f'df_new_plots_S{scenario}'][f'PowerLoss_S{scenario}'].std()

    mean_pl_si = scenarios[f'df_new_plots_S{scenario}_SI'][f'PowerLoss_S{scenario}_SI'].mean()
    std_pl_si = scenarios[f'df_new_plots_S{scenario}_SI'][f'PowerLoss_S{scenario}_SI'].std()

    mean_pl_di = scenarios[f'df_new_plots_S{scenario}_DI'][f'PowerLoss_S{scenario}_DI'].mean()
    std_pl_di = scenarios[f'df_new_plots_S{scenario}_DI'][f'PowerLoss_S{scenario}_DI'].std()

    # Añadir las estadísticas al gráfico (esquina superior izquierda)
    ax.text(0.05, 0.95, f'Mean BL: {mean_pl:.2f}\nStd: {std_pl:.2f}', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8), color=palette[0])

    ax.text(0.05, 0.80, f'Mean SC: {mean_pl_si:.2f}\nStd SI: {std_pl_si:.2f}', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8), color=palette[2])

    ax.text(0.05, 0.65, f'Mean DC: {mean_pl_di:.2f}\nStd DI: {std_pl_di:.2f}', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8), color=palette[4])

    # Mover la leyenda a la esquina superior derecha
    ax.legend(loc='upper right', fontsize=10)

    # Títulos y etiquetas por subplot
    ax.set_title(f'Scenario {scenario}', fontsize=14)
    ax.set_xlabel('PowerLoss', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)

# Ajustar la disposición para que los subplots no se solapen
plt.tight_layout()
plt.show()
#%% powerloss boxplots
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
plt.style.use('default')  # Usar estilo por defecto
%matplotlib inline

# Crear la figura y los ejes en formato de grid 2x3
fig, axs = plt.subplots(2, 3, figsize=(20, 10))

# Lista de escenarios
scenarios_list = ['0', '1', '2', '3', '4', '5']
palette = sns.color_palette("cubehelix", 3)  # Paleta de colores para tres categorías

# Iterar sobre los escenarios y sobre los ejes correspondientes
for idx, scenario in enumerate(scenarios_list):
    ax = axs[idx // 3, idx % 3]  # Seleccionar el eje correspondiente (2x3 grid)
    
    # Crear un DataFrame para los PowerLoss de cada tipo
    data = {
        'PowerLoss': pd.concat([
            scenarios[f'df_new_plots_S{scenario}'][f'PowerLoss_S{scenario}'],
            scenarios[f'df_new_plots_S{scenario}_SI'][f'PowerLoss_S{scenario}_SI'],
            scenarios[f'df_new_plots_S{scenario}_DI'][f'PowerLoss_S{scenario}_DI']
        ]).reset_index(drop=True),
        'Type': ['BL'] * len(scenarios[f'df_new_plots_S{scenario}'][f'PowerLoss_S{scenario}']) + \
                ['SC'] * len(scenarios[f'df_new_plots_S{scenario}_SI'][f'PowerLoss_S{scenario}_SI']) + \
                ['DC'] * len(scenarios[f'df_new_plots_S{scenario}_DI'][f'PowerLoss_S{scenario}_DI'])
    }
    
    df_powerloss = pd.DataFrame(data)
    
    # Graficar el boxplot
    sns.boxplot(x='Type', y='PowerLoss', data=df_powerloss, palette=palette, ax=ax)
    
    # Añadir títulos y etiquetas
    ax.set_title(f'Scenario {scenario}', fontsize=14)
    ax.set_xlabel('Type', fontsize=12)
    ax.set_ylabel('PowerLoss', fontsize=12)

# Ajustar la disposición para que los subplots no se solapen
plt.tight_layout()
plt.show()

#%% power loss diff in histogram
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
plt.style.use('default')  # Usar estilo por defecto
%matplotlib inline

# Crear la figura y los ejes en formato de grid 2x3
fig, axs = plt.subplots(2, 3, figsize=(20, 10))

# Lista de escenarios
scenarios_list = ['0', '1', '2', '3', '4', '5']
palette = sns.color_palette("cubehelix", 2)  # Paleta de colores para las diferencias

# Iterar sobre los escenarios y sobre los ejes correspondientes
for idx, scenario in enumerate(scenarios_list):
    ax = axs[idx // 3, idx % 3]  # Seleccionar el eje correspondiente (2x3 grid)
    
    # Obtener los datos de PowerLoss
    powerloss_bl = scenarios[f'df_new_plots_S{scenario}'][f'PowerLoss_S{scenario}']
    powerloss_si = scenarios[f'df_new_plots_S{scenario}_SI'][f'PowerLoss_S{scenario}_SI']
    powerloss_di = scenarios[f'df_new_plots_S{scenario}_DI'][f'PowerLoss_S{scenario}_DI']
    
    # Calcular las diferencias
    diff_bl_sc = powerloss_bl - powerloss_si
    diff_bl_dc = powerloss_bl - powerloss_di

    # Graficar los histogramas de las diferencias
    sns.histplot(diff_bl_sc, bins=30, color=palette[0], ax=ax, label='BL - SC', stat='frequency', kde=True, alpha=0.5)
    sns.histplot(diff_bl_dc, bins=30, color=palette[1], ax=ax, label='BL - DC', stat='frequency', kde=True, alpha=0.5)

    # Títulos y etiquetas
    ax.set_title(f'Scenario {scenario}', fontsize=14)
    ax.set_xlabel('Difference in PowerLoss', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.legend()

# Ajustar la disposición para que los subplots no se solapen
plt.tight_layout()
plt.show()


#%% powerloss power diff in boxplots 
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch

plt.style.use('default')  # Usar estilo por defecto
%matplotlib inline

# Crear la figura y los ejes en formato de grid 1x6 (1 fila y 6 columnas)
fig, axs = plt.subplots(1, 6, figsize=(20, 10))  # Tamaño ajustado para una fila y seis columnas

# Lista de escenarios
scenarios_list = ['0', '1', '2', '3', '4', '5']

# Obtener los colores [2] y [4] de la paleta cubehelix
palette = sns.color_palette("cubehelix")  # Genera 5 colores de la paleta
colors = [palette[2], palette[4]]  # Seleccionar los colores 2 y 4

# Inicializar listas para almacenar las diferencias
all_differences = []

# Iterar sobre los escenarios y sobre los ejes correspondientes
for idx, scenario in enumerate(scenarios_list):
    ax = axs[idx]  # Seleccionar el eje correspondiente (1 fila, 6 columnas)
    
    # Obtener los datos de PowerLoss
    powerloss_bl = scenarios[f'df_new_plots_S{scenario}'][f'PowerLoss_S{scenario}']
    powerloss_si = scenarios[f'df_new_plots_S{scenario}_SI'][f'PowerLoss_S{scenario}_SI']
    powerloss_di = scenarios[f'df_new_plots_S{scenario}_DI'][f'PowerLoss_S{scenario}_DI']
    
    # Calcular las diferencias
    max_powerloss_bl = powerloss_bl.max()
    diff_bl_sc = max_powerloss_bl - powerloss_si
    diff_bl_dc = powerloss_bl - powerloss_di

    # Crear un DataFrame para las diferencias
    data = {
        'Difference': pd.concat([diff_bl_sc, diff_bl_dc]).reset_index(drop=True),
        'Type': ['BL - SC'] * len(diff_bl_sc) + ['BL - DC'] * len(diff_bl_dc)
    }
    
    df_diff = pd.DataFrame(data)
    
    # Almacenar las diferencias para ajustar la escala
    all_differences.append(df_diff['Difference'])

    # Graficar el boxplot con los colores [2] y [4]
    sns.boxplot(x='Type', y='Difference', data=df_diff, palette=colors, ax=ax, width=0.3)
    
    # Calcular estadísticas descriptivas
    stats = df_diff.groupby('Type')['Difference'].describe()
    
    # Preparar el texto combinado para las estadísticas de BL - SC y BL - DC
    stats_text = (
        f"BL - SC  |  BL - DC\n"
        f"Min:    {stats.loc['BL - SC']['min']:.2f}  |  {stats.loc['BL - DC']['min']:.2f}\n"
        f"Q1:     {stats.loc['BL - SC']['25%']:.2f}  |  {stats.loc['BL - DC']['25%']:.2f}\n"
        f"Median: {stats.loc['BL - SC']['50%']:.2f}  |  {stats.loc['BL - DC']['50%']:.2f}\n"
        f"Q3:     {stats.loc['BL - SC']['75%']:.2f}  |  {stats.loc['BL - DC']['75%']:.2f}\n"
        f"Max:    {stats.loc['BL - SC']['max']:.2f}  |  {stats.loc['BL - DC']['max']:.2f}"
    )
       
    # Añadir el texto dentro del gráfico
    ax.text(0.5, - 230, stats_text, ha='center', va='top', fontsize=16,
            bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='lightgrey'))

    # Añadir títulos y etiquetas
    ax.set_title(f'Scenario {scenario}', fontsize=20)
    ax.set_xlabel('', fontsize=18)
    ax.set_ylabel('Contingency Difference [MW]', fontsize=20)

    # Ajustar el tamaño de los ticks en X e Y
    ax.tick_params(axis='both', which='major', labelsize=18)

# Ajustar la escala del eje y para todos los subplots
y_min = min([diff.min() for diff in all_differences])
y_max = max([diff.max() for diff in all_differences])
for ax in axs.flat:
    ax.set_ylim(y_min - 40, y_max + 20)  # Ajuste para dar más espacio para las estadísticas

# Ajustar la disposición para que los subplots no se solapen
plt.tight_layout()
plt.show()




#%% power % diff in boxplots
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

plt.style.use('default')  # Usar estilo por defecto
%matplotlib inline

# Crear la figura y los ejes en formato de grid 2x3
fig, axs = plt.subplots(2, 3, figsize=(20, 10))

# Lista de escenarios
scenarios_list = ['0', '1', '2', '3', '4', '5']
palette = sns.color_palette("cubehelix", 2)  # Paleta de colores para las diferencias

# Iterar sobre los escenarios y sobre los ejes correspondientes
for idx, scenario in enumerate(scenarios_list):
    ax = axs[idx // 3, idx % 3]  # Seleccionar el eje correspondiente (2x3 grid)
    
    # Obtener los datos de PowerLoss
    powerloss_bl = scenarios[f'df_new_plots_S{scenario}'][f'PowerLoss_S{scenario}']
    powerloss_si = scenarios[f'df_new_plots_S{scenario}_SI'][f'PowerLoss_S{scenario}_SI']
    powerloss_di = scenarios[f'df_new_plots_S{scenario}_DI'][f'PowerLoss_S{scenario}_DI']
    
    # Calcular las diferencias en porcentaje
    diff_bl_sc = ((powerloss_bl - powerloss_si) * 100) / powerloss_bl
    diff_bl_dc = ((powerloss_bl - powerloss_di) * 100) / powerloss_bl

    # Crear un DataFrame para las diferencias
    data = {
        'Difference': pd.concat([diff_bl_sc, diff_bl_dc]).reset_index(drop=True),
        'Type': ['BL - SC'] * len(diff_bl_sc) + ['BL - DC'] * len(diff_bl_dc)
    }
    
    df_diff = pd.DataFrame(data)
    
    # Graficar el boxplot
    sns.boxplot(x='Type', y='Difference', data=df_diff, palette=palette, ax=ax)
    
    # Calcular estadísticas descriptivas
    stats = df_diff.groupby('Type')['Difference'].describe()
    
    # Añadir estadísticas al gráfico
    for i, (type_label, stat_row) in enumerate(stats.iterrows()):
        median = stat_row['50%']
        q1 = stat_row['25%']
        q3 = stat_row['75%']
        ax.text(i, median + 0.5, f'Median: {median:.2f}%', ha='center', va='bottom', fontsize=10, color='black')
        ax.text(i, q1 - 0.5, f'Q1: {q1:.2f}%', ha='center', va='top', fontsize=10, color='black')
        ax.text(i, q3 + 0.5, f'Q3: {q3:.2f}%', ha='center', va='bottom', fontsize=10, color='black')
    
    # Añadir títulos y etiquetas
    ax.set_title(f'Scenario {scenario}', fontsize=14)
    ax.set_xlabel('Difference in PowerLoss (%)', fontsize=12)
    ax.set_ylabel('Value (%)', fontsize=12)

# Ajustar la disposición para que los subplots no se solapen
plt.tight_layout()
plt.show()


#%%
# PLOTS DE INERCIA CON MATPLOTLIB

# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Lista de escenarios
# # sce_list = ['S0', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6']
# # sce_list = ['S0_SI', 'S1_SI', 'S2_SI', 'S3_SI', 'S4_SI', 'S5_SI', 'S6_SI']
# # sce_list = ['S0_DI', 'S1_DI', 'S2_DI', 'S3_DI', 'S4_DI', 'S5_DI', 'S6_DI']
# sce_list = ['S6','S6_DI','S6_SI']

# # Crear un índice de tiempo común (8760 horas)
# common_index = pd.date_range(start='2030-01-01', periods=8760, freq='H')

# # Ajustar los DataFrames al mismo rango temporal y seleccionar las columnas relevantes
# combined_df_list = []
# for s in sce_list:
#     # Resetear y reindexar al rango común
#     df = scenarios[f'df_new_plots_{s}'].set_index('TIMESTAMP').reindex(common_index)
    
#     # Renombrar columnas para mantenerlas consistentes en todos los DataFrames
#     df = df.rename(columns={f'TRG_{s}': 'VREGeneration', f'TCONV_{s}': 'ConventionalGeneration', f'PL_{s}': 'PenetrationLevel', f'Curtailment_{s}': 'Curtailment', f'SystemInertia_{s}': 'SystemInertia', f'SystemGain_{s}': 'SystemGain', f'PowerLoss_{s}': 'PowerLoss', f'InertiaLimit_{s}': 'InertiaLimit'})
    
#     # Añadir una columna para identificar el escenario
#     df['Scenario'] = s
    
#     # Agregar al listado de DataFrames a combinar
#     combined_df_list.append(df)

# # Combinar todos los DataFrames en uno solo
# combined_df = pd.concat(combined_df_list,axis=0)

# # Restablecer el índice para usar 'TIMESTAMP' como columna
# combined_df = combined_df.reset_index().rename(columns={'index': 'TIMESTAMP'})

# # Definir el rango de tiempo deseado (ajusta estas fechas según lo que necesites)
# start_date = '2030-06-01'
# end_date = '2030-06-15'
# mask = (combined_df['TIMESTAMP'] >= start_date) & (combined_df['TIMESTAMP'] <= end_date)

# # Filtrar el DataFrame para incluir solo los datos dentro del rango de tiempo especificado
# filtered_df = combined_df[mask]

# # Configurar el tamaño del gráfico
# plt.figure(figsize=(150, 100))  # Ancho de 20, altura de 30

# # Iterar a través de los escenarios y crear un subgráfico para cada uno
# for i, scenario in enumerate(sce_list):
#     plt.subplot(len(sce_list), 1, i + 1)  # Un gráfico por fila
    
#     # Filtrar datos para el escenario actual
#     scenario_data = filtered_df[filtered_df['Scenario'] == scenario]
    
#     # Graficar el área para 'SystemInertia'
#     plt.fill_between(scenario_data['TIMESTAMP'], scenario_data['SystemInertia'], color='teal', alpha=0.5, label='SystemInertia')
    
#     # Graficar una línea segmentada para 'InertiaLimit'
#     plt.plot(scenario_data['TIMESTAMP'], scenario_data['InertiaLimit'], color='red', alpha=1, label='InertiaLimit')
    
#     # Ajustar etiquetas y leyendas
#     plt.xlabel('TIMESTAMP', fontsize=80)
#     plt.ylabel('Inertia [s]', fontsize=80)
#     plt.title(f'Scenario: {scenario}', fontsize=100)
#     plt.legend(loc='upper right', fontsize=60)
    
#     # Mejorar la resolución del eje x y ajustar las etiquetas
#     plt.xticks(pd.date_range(start=start_date, end=end_date, freq='D'), rotation=45, fontsize=80)  # Mostrar etiquetas diarias
#     plt.yticks(fontsize=80)

# # Ajustar el diseño para evitar superposición
# plt.tight_layout()

# # Mostrar los gráficos
# plt.show()

#%%
# # PLOTS DE INERCIA CON SEABORN
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt


# # Lista de escenarios
# sce_list = ['S0', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6']
# # sce_list = ['S0_SI', 'S1_SI', 'S2_SI', 'S3_SI', 'S4_SI', 'S5_SI', 'S6_SI']
# # sce_list = ['S0_DI', 'S1_DI', 'S2_DI', 'S3_DI', 'S4_DI', 'S5_DI', 'S6_DI']

# # Crear un índice de tiempo común (8760 horas)
# common_index = pd.date_range(start='2030-01-01', periods=8760, freq='H')

# # Ajustar los DataFrames al mismo rango temporal y seleccionar las columnas relevantes
# combined_df_list = []
# for s in sce_list:
#     # Resetear y reindexar al rango común
#     df = scenarios[f'df_new_plots_{s}'].set_index('TIMESTAMP').reindex(common_index)
    
#     # Renombrar columnas para mantenerlas consistentes en todos los DataFrames
#     df = df.rename(columns={f'SystemInertia_{s}': 'SystemInertia', f'InertiaLimit_{s}': 'InertiaLimit'})
    
#     # Añadir una columna para identificar el escenario
#     df['Scenario'] = s
    
#     # Agregar al listado de DataFrames a combinar
#     combined_df_list.append(df)

# # Combinar todos los DataFrames en uno solo
# combined_df = pd.concat(combined_df_list)

# # Restablecer el índice para usar 'TIMESTAMP' como columna
# combined_df = combined_df.reset_index().rename(columns={'index': 'TIMESTAMP'})

# # Definir el rango de tiempo deseado
# start_date = '2030-06-01'
# end_date = '2030-06-15'
# mask = (combined_df['TIMESTAMP'] >= start_date) & (combined_df['TIMESTAMP'] <= end_date)

# # Filtrar el DataFrame para incluir solo los datos dentro del rango de tiempo especificado
# filtered_df = combined_df[mask]

# # Configurar el estilo de Seaborn
# sns.set_style("whitegrid")
# sns.set_context("talk", font_scale=1.5)

# # Crear un FacetGrid con Seaborn
# g = sns.FacetGrid(filtered_df, row='Scenario', height=4, aspect=4, sharex=True, sharey=False)

# # Graficar los datos en cada subplot
# def plot_data(data, **kwargs):
#     ax = plt.gca()
#     # Rellenar el área para 'SystemInertia'
#     ax.fill_between(data['TIMESTAMP'], data['SystemInertia'], color='teal', alpha=0.5, label='SystemInertia')
#     # Graficar las barras para 'SystemInertia'
#     ax.bar(data['TIMESTAMP'], data['SystemInertia'], color='teal', alpha=0.3, width=0.01)
#     # Graficar una línea segmentada para 'InertiaLimit'
#     sns.lineplot(data=data, x='TIMESTAMP', y='InertiaLimit', color='red', alpha=1, ax=ax, label='InertiaLimit')

# # Mapear la función personalizada a cada FacetGrid
# g.map_dataframe(plot_data)

# # Ajustar etiquetas y títulos
# g.set_axis_labels('TIMESTAMP', 'Inertia [s]')
# g.set_titles('Scenario: {row_name}', fontsize=20)
# g.add_legend()

# # Ajustar el formato de las etiquetas del eje x
# for ax in g.axes.flat:
#     ax.set_xticks(pd.date_range(start=start_date, end=end_date, freq='D'))
#     ax.set_xticklabels(pd.date_range(start=start_date, end=end_date, freq='D').strftime('%Y-%m-%d'), rotation=45)

# # Ajustar la leyenda
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=15)

# # Mostrar el gráfico
# plt.tight_layout()
# plt.show()


#%%
# PLOTS DE CURTAILMENT CON MATPLOTLIB

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Lista de escenarios
sce_list = ['S0', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6']
# sce_list = ['S0_SI', 'S1_SI', 'S2_SI', 'S3_SI', 'S4_SI', 'S5_SI', 'S6_SI']
# sce_list = ['S0_DI', 'S1_DI', 'S2_DI', 'S3_DI', 'S4_DI', 'S5_DI', 'S6_DI']


# Crear un índice de tiempo común (8760 horas)
common_index = pd.date_range(start='2030-01-01', periods=8760, freq='H')

# Ajustar los DataFrames al mismo rango temporal y seleccionar las columnas relevantes
combined_df_list = []
for s in sce_list:
    # Resetear y reindexar al rango común
    df = scenarios[f'df_new_plots_{s}'].set_index('TIMESTAMP').reindex(common_index)
    
    # Renombrar columnas para mantenerlas consistentes en todos los DataFrames
    df = df.rename(columns={f'Curtailment_{s}': 'Curtailment'})
    
    # Añadir una columna para identificar el escenario
    df['Scenario'] = s
    
    # Agregar al listado de DataFrames a combinar
    combined_df_list.append(df)

# Combinar todos los DataFrames en uno solo
combined_df = pd.concat(combined_df_list)

# Restablecer el índice para usar 'TIMESTAMP' como columna
combined_df = combined_df.reset_index().rename(columns={'index': 'TIMESTAMP'})

# Definir el rango de tiempo deseado (ajusta estas fechas según lo que necesites)
start_date = '2030-01-01'
end_date = '2030-12-31'
mask = (combined_df['TIMESTAMP'] >= start_date) & (combined_df['TIMESTAMP'] <= end_date)

# Filtrar el DataFrame para incluir solo los datos dentro del rango de tiempo especificado
filtered_df = combined_df[mask]

# Configurar el tamaño del gráfico
plt.figure(figsize=(150, 100))  # Ancho de 20, altura de 30

# Iterar a través de los escenarios y crear un subgráfico para cada uno
for i, scenario in enumerate(sce_list):
    plt.subplot(len(sce_list), 1, i + 1)  # Un gráfico por fila
    
    # Filtrar datos para el escenario actual
    scenario_data = filtered_df[filtered_df['Scenario'] == scenario]
    
    # Graficar el área para 'curtailment'
    plt.fill_between(scenario_data['TIMESTAMP'], scenario_data['Curtailment'], color='darkgreen', alpha=0.5, label='Curtailment')
    
    # Ajustar etiquetas y leyendas
    plt.xlabel('TIMESTAMP', fontsize=80)
    plt.ylabel('Curtailment [MWh]', fontsize=80)
    plt.title(f'Scenario: {scenario}', fontsize=100)
    plt.legend(loc='upper right', fontsize=60)
    
    # Mejorar la resolución del eje x y ajustar las etiquetas
    plt.xticks(pd.date_range(start=start_date, end=end_date, freq='M'), rotation=45, fontsize=80)  # Mostrar etiquetas diarias
    plt.yticks(fontsize=80)

# Ajustar el diseño para evitar superposición
plt.tight_layout()

# Mostrar los gráficos
plt.show()

#%%


list_to_plot=['VREGeneration', 'ConventionalGeneration', 'PenetrationLevel', 'Curtailment', 'SystemInertia', 'SystemGain', 'PowerLoss']

for i in list_to_plot:
	fig, ax = plt.subplots(figsize=(14,7), )
	sns.kdeplot(data = combined_df, x=i, hue = 'Scenario', fill=True, alpha=.5, linewidth=0, palette = 'viridis')
	plt.show()
    
    
for i in list_to_plot:
	fig, ax = plt.subplots(figsize=(14,7), )
	sns.jointplot(data = combined_df, y='SystemInertia', x = i, hue = 'Scenario', alpha=.5, linewidth=0, palette = 'viridis')
	plt.show()    







# Restablecer el índice para usar 'TIMESTAMP' como columna
combined_df = combined_df.reset_index().rename(columns={'index': 'TIMESTAMP'})

# Definir el rango de tiempo deseado
start_date = '2030-01-01'
end_date = '2030-12-31'
mask = (combined_df['TIMESTAMP'] >= start_date) & (combined_df['TIMESTAMP'] <= end_date)

# Filtrar el DataFrame para incluir solo los datos dentro del rango de tiempo especificado
filtered_df = combined_df[mask]

for i, scenario in enumerate(sce_list):
    plt.subplot(len(sce_list), 1, i + 1)  # Un gráfico por fila
    
    # Filtrar datos para el escenario actual
    scenario_data = filtered_df[filtered_df['Scenario'] == scenario]
        
    from statsmodels.tsa.seasonal import STL
    for i in list_to_plot:
    	plt.rc('figure', figsize=(16,12))
    	plt.rc('font', size=10)
    	Y = df[i].fillna(0)
    	stl = STL(Y)
    	res = stl.fit()
    	fig = res.plot()






# from statsmodels.graphics.gofplots import qqplot
# for i in list_to_plot:
# 	Y = filtered_df[i].fillna(0)
# 	X = filtered_df_diff[i].fillna(0)
# 	plt.rc('figure', figsize=(5,5), )
# 	#stl = STL(Y)
# 	#res = stl.fit()
# 	qqplot(Y, line='q')
# 	#qqplot(res.resid, line='q')
# 	qqplot(X, line='q')
# 	plt.show()
# 	print(i)
    
    
    
    
import pandas as pd

# Supongamos que tu DataFrame se llama combined_df
combined_df['Hour'] = pd.to_datetime(combined_df['TIMESTAMP']).dt.hour
import seaborn as sns
import matplotlib.pyplot as plt

list_to_plot = ['VREGeneration', 'ConventionalGeneration', 'PenetrationLevel', 'Curtailment', 'SystemInertia', 'SystemGain', 'PowerLoss']

for i in list_to_plot:
    g = sns.FacetGrid(combined_df, col="Hour", hue="Scenario", col_wrap=4, palette='viridis', height=4)
    g.map(sns.kdeplot, i, fill=True, alpha=.5, linewidth=0)
    g.add_legend()
    plt.show()
    
    
    
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Convertir el timestamp a horas
combined_df['Hour'] = pd.to_datetime(combined_df['TIMESTAMP']).dt.hour

# Listado de variables a graficar
list_to_plot = ['VREGeneration', 'ConventionalGeneration', 'PenetrationLevel', 'Curtailment', 'SystemInertia', 'SystemGain', 'PowerLoss']

# Filtrar las primeras 12 horas (de 0 a 11)
df_first_12_hours = combined_df[combined_df['Hour'] < 12]

# Filtrar las siguientes 12 horas (de 12 a 23)
df_second_12_hours = combined_df[(combined_df['Hour'] >= 12) & (combined_df['Hour'] < 24)]

# Crear los gráficos para las primeras 12 horas
for i in list_to_plot:
    g = sns.FacetGrid(df_first_12_hours, col="Hour", hue="Scenario", col_wrap=4, palette='viridis', height=4)
    g.map(sns.kdeplot, i, fill=True, alpha=.5, linewidth=0)
    g.add_legend()
    plt.show()

# Crear los gráficos para las siguientes 12 horas
for i in list_to_plot:
    g = sns.FacetGrid(df_second_12_hours, col="Hour", hue="Scenario", col_wrap=4, palette='viridis', height=4)
    g.map(sns.kdeplot, i, fill=True, alpha=.5, linewidth=0)
    g.add_legend()
    plt.show()



#%%

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Lista de escenarios
# sce_list = ['S0', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6']
# sce_list = ['S0_SI', 'S1_SI', 'S2_SI', 'S3_SI', 'S4_SI', 'S5_SI', 'S6_SI']
# sce_list = ['S0_DI', 'S1_DI', 'S2_DI', 'S3_DI', 'S4_DI', 'S5_DI', 'S6_DI']
sce_list = ['S6','S6_DI','S6_SI']

# Crear un índice de tiempo común (8760 horas)
common_index = pd.date_range(start='2030-01-01', periods=8760, freq='H')

# Ajustar los DataFrames al mismo rango temporal y seleccionar las columnas relevantes
combined_df_list = []
for s in sce_list:
    # Resetear y reindexar al rango común
    df = scenarios[f'df_new_plots_{s}'].set_index('TIMESTAMP').reindex(common_index)
    
    # # Renombrar columnas para mantenerlas consistentes en todos los DataFrames
    # df = df.rename(columns={f'TRG_{s}': 'VREGeneration', f'TCONV_{s}': 'ConventionalGeneration', f'PL_{s}': 'PenetrationLevel', f'Curtailment_{s}': 'Curtailment', f'SystemInertia_{s}': 'SystemInertia', f'SystemGain_{s}': 'SystemGain', f'PowerLoss_{s}': 'PowerLoss', f'InertiaLimit_{s}': 'InertiaLimit'})
    
    # Añadir una columna para identificar el escenario
    df['Scenario'] = s
    
    # Agregar al listado de DataFrames a combinar
    combined_df_list.append(df)

# Combinar todos los DataFrames en uno solo
combined_df2 = pd.concat(combined_df_list,axis=1)


#%%distribucion de capacidad instalada en porcentajes por tecnologias


import matplotlib.pyplot as plt
import numpy as np

# Datos de los años y capacidades por tipo de generación
años = ["2023", "2025", "2030"]
capacidad_total = [3472.42, 4003.14, 4529.14]
termica = [2441.09, 2441.09, 2441.09]
hidroelectrica = [734.85, 1221.42, 1221.42]
eolica = [131.4, 175.55, 391.55]
solar = [165.08, 165.08, 475.08]

# Cálculo de porcentajes y etiquetas combinadas
termica_label = [f"{t} MW ({(t/c)*100:.2f}%)" for t, c in zip(termica, capacidad_total)]
hidroelectrica_label = [f"{h} MW ({(h/c)*100:.2f}%)" for h, c in zip(hidroelectrica, capacidad_total)]
eolica_label = [f"{e} MW ({(e/c)*100:.2f}%)" for e, c in zip(eolica, capacidad_total)]
solar_label = [f"{s} MW ({(s/c)*100:.2f}%)" for s, c in zip(solar, capacidad_total)]

# Apilar los datos para crear el gráfico de barras apiladas
barWidth = 0.5
r = np.arange(len(años))

plt.figure(figsize=(12, 8))

# Crear las barras apiladas
bars1 = plt.bar(r, termica, color='#ffb74d', edgecolor='gray', width=barWidth, label='Térmica')
bars2 = plt.bar(r, hidroelectrica, bottom=termica, color='#4fc3f7', edgecolor='gray', width=barWidth, label='Hidroeléctrica')
bars3 = plt.bar(r, eolica, bottom=np.array(termica)+np.array(hidroelectrica), color='#81c784', edgecolor='gray', width=barWidth, label='Eólica')
bars4 = plt.bar(r, solar, bottom=np.array(termica)+np.array(hidroelectrica)+np.array(eolica), color='#ffd54f', edgecolor='gray', width=barWidth, label='Solar')

# Etiquetas y título en español
plt.xlabel("Año", fontsize=12)
plt.ylabel("Capacidad instalada (MW)", fontsize=12)
plt.title("Capacidad Instalada del SIN en Bolivia por Año y Tipo de Generación", fontsize=14)
plt.xticks(r, años)
plt.legend(title="Tipo de Generación", fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Agregar valores en MW y % en cada barra
for i, rect in enumerate(bars1):
    plt.text(rect.get_x() + rect.get_width() / 2, rect.get_height() / 2, 
             termica_label[i], ha='center', va='center', color="black", fontsize=10)

for i, rect in enumerate(bars2):
    plt.text(rect.get_x() + rect.get_width() / 2, termica[i] + rect.get_height() / 2, 
             hidroelectrica_label[i], ha='center', va='center', color="black", fontsize=10)

for i, rect in enumerate(bars3):
    plt.text(rect.get_x() + rect.get_width() / 2, termica[i] + hidroelectrica[i] + rect.get_height() / 2, 
             eolica_label[i], ha='center', va='center', color="black", fontsize=10)

for i, rect in enumerate(bars4):
    plt.text(rect.get_x() + rect.get_width() / 2, termica[i] + hidroelectrica[i] + eolica[i] + rect.get_height() / 2, 
             solar_label[i], ha='center', va='center', color="black", fontsize=10)

plt.show()