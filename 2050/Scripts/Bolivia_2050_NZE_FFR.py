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

sys.path.append(os.path.abspath('..'))

# Import Dispa-SET
import dispaset as ds

# Load the configuration file
config = ds.load_config('../ConfigFiles/Config_BOLIVIA_2050_NZE_FFR.xlsx')

# # Limit the simulation period (for testing purposes, comment the line to run the whole year)
# config['StartDate'] = (2050, 1, 1, 0, 0, 0)
# config['StopDate'] = (2050, 1, 3, 0, 0, 0)

# # Build the simulation environment:
# # SimData = ds.build_simulation(config)
# SimData = ds.build_simulation(config,mts_plot=True,MTSTimeStep=24)

# # # Solve using GAMS:
# _ = ds.solve_GAMS(config['SimulationDirectory'], config['GAMS_folder'])

# Load the simulation results:
inputs,results = ds.get_sim_results(path='../Simulations/BOLIVIA_2050_NZE_FFR',cache=False)
# inputs,results = ds.get_sim_results(path='../Simulations/BOLIVIA_POWERFLOWDC_TEST',cache=False, inputs_file='Inputs_MTS.p',results_file='Results_MTS.gdx')
# inputs, results = ds.get_sim_results(config, cache=False)
inputs_MTS, results_MTS = ds.get_sim_results(path='../Simulations/BOLIVIA_2050_NZE_FFR', cache=False, inputs_file='Inputs_MTS.p',results_file='Results_MTS.gdx')

# # import pandas as pd
import pandas as pd

# Generate country-specific plots
# ds.plot_zone(inputs, results, rng=rng)
# rng = pd.date_range('2026-01-01', '2026-01-07', freq='H')
# for i in config['zones']:
#     ds.plot_zone(inputs, results, z=i, rng=rng)
    
rng = pd.date_range('2050-01-01', '2050-01-07', freq='H')
for i in config['zones']:
    ds.plot_zone(inputs, results, z=i, rng=rng)
    
# rng1 = pd.date_range('2050-06-01', '2050-06-07', freq='D')
# for j in config['zones']:
#     ds.plot_zone(inputs_MTS, results_MTS, z=j, rng=rng1)

# # Bar plot with the installed capacities in all countries:
# cap = ds.plot_zone_capacities(inputs, results)

# # Bar plot with installed storage capacity
# sto = ds.plot_tech_cap(inputs)

# # Violin plot for CO2 emissions
# ds.plot_co2(inputs, results, figsize=(9, 6), width=0.9)

# # Bar plot with the energy balances in all countries:
# ds.plot_energy_zone_fuel(inputs, results, ds.get_indicators_powerplant(inputs, results))

# Analyse the results for each country and provide quantitative indicators:
r = ds.get_result_analysis(inputs, results)

# # Analyze power flow tracing
# pft, pft_prct = ds.plot_power_flow_tracing_matrix(inputs, results, cmap="magma_r", figsize=(15, 10))

# # Plot net flows on a map
# ds.plot_net_flows_map(inputs, results, terrain=True, margin=3, bublesize=5000, figsize=(8, 7))

# # Plot congestion in the interconnection lines on a map
# ds.plot_line_congestion_map(inputs, results, terrain=True, margin=3, figsize=(9, 7), edge_width=3.5, bublesize=100)

# # Analyse the results of the dispatch and provide frequency security constraints:
# freq_sec_const, summary, dispatch = ds.get_frequency_security_constraints(inputs, results)

# # Save the results of the frequency security constraints analisys
# summary.to_csv('../Simulations/BOLIVIA_S1/Summary.csv', index=False)
# Contingency.to_csv('../Simulations/BOLIVIA_S1/Contingency.csv', index=False)
# with pd.ExcelWriter('../Simulations/BOLIVIA_S1/frec_sec_const.xlsx', engine='xlsxwriter') as writer:
#     for contingency, df in freq_sec_const.items():
#         df.to_excel(writer, sheet_name=contingency, index=False)
        
# # Read the results of the frequency security constraints analisys   
# summary = pd.read_csv('../Simulations/BOLIVIA_S1/summary.csv')  
# dispatch = pd.read_csv('../Simulations/BOLIVIA_S1/dispatch.csv')     
# freq_sec_const = pd.read_excel('../Simulations/BOLIVIA_S1/frec_sec_const.xlsx', sheet_name=None)
# %% ESTO ES PARA ENCONTRAR LAS COMBINACIONES POSIBLES DE INERCIA, Y GANANCIA PARA LOS VALORES POSIBLES DE COMMITTED
# import itertools
# import numpy as np
# import pandas as pd

# # Extract relevant data
# conventional_units = inputs["sets"]["cu"]

# # Filter PowerCapacity, InertiaConstant, and Droop to include only conventional units
# filtered_power_capacity = inputs["param_df"]["PowerCapacity"].loc[conventional_units]
# filtered_inertia_constant = inputs["param_df"]["InertiaConstant"].loc[conventional_units]
# filtered_droop = inputs["param_df"]["Droop"].loc[conventional_units]

# # Convert to numpy arrays for computation
# power_capacity = filtered_power_capacity["PowerCapacity"].to_numpy()
# inertia_constant = filtered_inertia_constant["InertiaConstant"].to_numpy()
# droop = filtered_droop["Droop"].to_numpy()

# # Nominal frequency
# f0 = 50  # Example nominal frequency (Hz)

# # Generate all possible combinations of commitment statuses
# all_combinations = itertools.product([0, 1], repeat=len(conventional_units))

# # Compute results
# system_inertia_values = []
# system_gain_values = []
# for combination in all_combinations:
#     combination_array = np.array(combination)

#     # Calculate System Inertia
#     system_inertia = np.sum(power_capacity * inertia_constant * combination_array)

#     # Calculate System Gain
#     system_gain = np.sum((power_capacity * combination_array) / (droop * f0))

#     # Store the pair (SystemInertia, SystemGain)
#     results.append((system_inertia, system_gain))

# # Sort results by SystemInertia
# sorted_results = sorted(results, key=lambda x: x[0])

# # Separate sorted values into individual lists
# sorted_inertia_values = [r[0] for r in sorted_results]
# sorted_gain_values = [r[1] for r in sorted_results]

# # Display results
# print("Sorted SystemInertia values:")
# print(sorted_inertia_values)

# print("SystemGain values sorted by SystemInertia:")
# print(sorted_gain_values)


# %%

OutputPower = results['OutputPower'] 
units = inputs['units']
            
# Iterar sobre las filas del DataFrame OutputPower
Contingency = pd.DataFrame(index=OutputPower.index)
for index, row in OutputPower.iterrows():
    # Encontrar la columna con el valor máximo en la fila actual
    max_column = row.idxmax()

    # Obtener el valor máximo
    max_value = row[max_column]
        
    # Agregar el valor máximo y la columna correspondiente al DataFrame de resultados
    Contingency.at[index, 'GeneratorName'] = max_column
    Contingency.at[index, 'Contingency[MW]'] = max_value*-1
    
    # Calcular la suma de toda la fila y restarle el valor máximo
    tot_out_pow = row.sum() - max_value
    Contingency.at[index, 'TotalOutputPower[MW]'] = tot_out_pow

Contingency = Contingency.reset_index()
Contingency.rename(columns={'index': 'TIMESTAMP'}, inplace=True)


# Find the max Contingnecy for each GeneratorName
max_ploss = Contingency.groupby('GeneratorName')['Contingency[MW]'].max()



# %%

import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt

counter = 1  # Inicializar el contador de hojas
total_contingencies = len(Contingency)  # Contar el número total de contingencias
print(f"Total Contingencies: {total_contingencies}")
# Create an empty dictionary to store results
freq_security = {}

# Create an Excel file to store the results
with pd.ExcelWriter('power_swing_results.xlsx', engine='xlsxwriter') as writer:
    contingency_count = 1  # Initialize the sheet counter

    for index, row in Contingency.iterrows():
        # Perform the operations on each row

        def power_swing(y, t, k1, k, H):
            deltap, f = y

            if t < 1:
                contingency = 0
                primaryreserve = 0
                ffr = 0
                deltap = contingency
            elif t < time_delay:
                contingency = row['Contingency[MW]'] 
                primaryreserve = 0
                ffr = 0
                deltap = contingency
            elif t < time_preparation:
                contingency = row['Contingency[MW]']
                primaryreserve = 0
                ffr = (-k1 * f) * (t-time_delay)
                deltap = contingency + ffr 
            else:
                contingency = row['Contingency[MW]'] 
                primaryreserve = (-k * f) * (t-time_preparation)
                ffr = (-k1 * f) * (t-time_delay)
                deltap = contingency + ffr + primaryreserve
            
            delf = deltap*50 / (2 * H *1000)
            return [deltap, delf]

        # Initial conditions
        f0 = 0
        y0 = [0, f0]

        # Time vector
        t = np.arange(0, 61, 0.1)  # Time steps every 1 second from 0 to 60 seconds

        # Create an empty DataFrame to store results
        results_df = pd.DataFrame({'Time': t})

        # Function to calculate frequency at each time step
        def calc_frequency_at_time(k1, k, H):
            sol = odeint(power_swing, y0, t, args=(k1, k, H))
            f_values = sol[:, 1]  # Frequency values at each time step
            return f_values

        # Constraints
        def constraints_satisfied(k1, k, H):
            f_values = calc_frequency_at_time(k1, k, H)
            der_f_values = np.gradient(f_values, t)
            min_der_f = np.min(der_f_values)
            min_f = np.min(f_values)
            f_at_60s = f_values[-1]

            # return min_der_f >= -0.004 and min_f >= -0.016 and f_at_60s >= -0.01
            # return min_der_f >= -0.2 and min_f >= -0.8 and f_at_60s >= -0.5
            return min_der_f >= -0.5 and min_f >= -0.8 and f_at_60s >= -0.5
              
        # Set the constant and time preparation values
        found_solution = False
        min_k = None                                  
        min_k1 = None
        min_H = None
        time_delay = 2
        time_preparation = 5       


       #######################################################################
              
       #  Adjust Initial bounds as of the parameters as needed     
        lower_bound_H = 5
        upper_bound_H = 37
        step_H = 1
        lower_bound_k = 100
        upper_bound_k = 3150
        step_k = 50
        lower_bound_k1 = 200
        upper_bound_k1 = 7500 
        step_k1 = 200     
        
       # Nested binary search
        def binary_search_k1(k, H, lower_bound_k1, upper_bound_k1, step_k1):
            """
            Realiza una búsqueda binaria sobre `k1` para valores fijos de `k` y `H`.
            Devuelve el valor de `k1` que satisface las restricciones o None si no existe.
            """
            solution_k1 = None
            while lower_bound_k1 <= upper_bound_k1:
                mid_k1 = (lower_bound_k1 + upper_bound_k1) // 2
                if constraints_satisfied(mid_k1, k, H):
                    upper_bound_k1 = mid_k1 - step_k1  # Intenta encontrar un k1 más pequeño
                    solution_k1 = mid_k1
                else:
                    lower_bound_k1 = mid_k1 + step_k1
            return solution_k1
        
        
        def binary_search_k(H, lower_bound_k, upper_bound_k, step_k, lower_bound_k1, upper_bound_k1, step_k1):
            """
            Realiza una búsqueda binaria sobre `k`, anidando la búsqueda de `k1`.
            Devuelve los valores óptimos de `k` y `k1` que satisfacen las restricciones.
            """
            solution_k = None
            solution_k1 = None
            while lower_bound_k <= upper_bound_k:
                mid_k = (lower_bound_k + upper_bound_k) // 2
                current_k1 = binary_search_k1(mid_k, H, lower_bound_k1, upper_bound_k1, step_k1)
                if current_k1 is not None:
                    upper_bound_k = mid_k - step_k  # Intenta reducir k
                    solution_k = mid_k
                    solution_k1 = current_k1
                else:
                    lower_bound_k = mid_k + step_k
            return solution_k, solution_k1
        
        
        def binary_search_H(lower_bound_H, upper_bound_H, step_H, lower_bound_k, upper_bound_k, step_k, lower_bound_k1, upper_bound_k1, step_k1):
            """
            Realiza una búsqueda binaria sobre `H`, anidando las búsquedas de `k` y `k1`.
            Devuelve los valores óptimos de `H`, `k` y `k1` que satisfacen las restricciones.
            """
            solution_H = None
            solution_k = None
            solution_k1 = None
            found_solution = False
        
            while lower_bound_H <= upper_bound_H:
                mid_H = (lower_bound_H + upper_bound_H) // 2
                current_k, current_k1 = binary_search_k(mid_H, lower_bound_k, upper_bound_k, step_k, lower_bound_k1, upper_bound_k1, step_k1)
                if current_k is not None and current_k1 is not None:
                    upper_bound_H = mid_H - step_H  # Intenta reducir H
                    solution_H = mid_H
                    solution_k = current_k
                    solution_k1 = current_k1
                else:
                    lower_bound_H = mid_H + step_H
        
            if solution_H is not None and solution_k is not None and solution_k1 is not None:
                found_solution = True
        
            return solution_H, solution_k, solution_k1, found_solution


        optimal_H, optimal_k, optimal_k1, found_solution = binary_search_H(lower_bound_H, 
        upper_bound_H, step_H, lower_bound_k, upper_bound_k, step_k, lower_bound_k1, upper_bound_k1, step_k1)
         
       #######################################################################
        
        if found_solution:
            print(f"Contingency {counter}")
            print(f"The delay time for the FFR is set at: {time_delay}")
            print(f"The preparation time for the Primary Reserve is set at: {time_preparation}")
            print(f"Minimum k1 that satisfies constraints: {optimal_k1}")# k variable
            print(f"Minimum k that satisfies constraints: {optimal_k}")# k variable
            print(f"Minimum H that satisfies constraints: {optimal_H}")
            # Update the DataFrame with the new values
            Contingency.at[index, 'k1'] = optimal_k1  # Store in 'k1'
            Contingency.at[index, 'k'] = optimal_k  # Store in 'k'
            Contingency.at[index, 'min_H'] = optimal_H  # Store in 'min_H'

            # Solve the ODE system with the optimal k and H
            sol = odeint(power_swing, y0, t, args=(optimal_k1, optimal_k, optimal_H))

            # Extract the results
            # deltap = sol[:, 0]
            f = sol[:, 1]
            der_f = np.gradient(f, t)
            ffr = np.where(t < time_delay, 0, ((-optimal_k1 * f) * (t-time_delay)))
            primaryreserve = np.where(t < time_preparation, 0, ((-optimal_k * f) * (t-time_preparation)))
            contingency = np.where(t < 1, 0, row['Contingency[MW]'])
            deltap = np.where(t < 1, 0, contingency + ffr + primaryreserve)
            totaloutputpower = row['TotalOutputPower[MW]']

            # Store results in the DataFrame
            results_df['TIMESTAMP'] = row['TIMESTAMP']
            results_df['Frequency[Hz]'] = f
            results_df['RoCoF[Hz/s]'] = der_f
            results_df['DeltaP[MW]'] = deltap
            results_df['Contingency[MW]'] = contingency
            results_df['PrimaryReserve[MW]'] = primaryreserve
            results_df['FFR[MW]'] = ffr

            # Add the values of k and H to the DataFrame
            results_df['min_K1[GW/Hz]'] = optimal_k1
            results_df['min_K[GW/Hz]'] = optimal_k
            results_df['min_H[GW*s]'] = optimal_H

            # Store the results_df in the dictionary
            freq_security[f"Contingency{contingency_count}"] = results_df    

            # Guardar el DataFrame en una pestaña numerada
            sheet_name = f'Contingency{contingency_count}'
            results_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            contingency_count += 1  # Incrementar el contador de hojas
           ####################################################################### 
            # # PLOT IN NORMAL UNITS
            # # Plotting code remains the same
            # fig, ax1 = plt.subplots(figsize=(12, 8))
        
            # # # Frequency Values Plot (Left Y-axis)
            # ax1.plot(t, f, color='#1f77b4', linestyle='--', label='$∆f_{COI}$')
            # ax1.plot(t, der_f, color='#ff7f0e', linestyle='--', label='$d(∆f_{COI})$')
            # ax1.set_xlabel('Time [s]', fontsize=16)
            # ax1.set_ylabel('Frequency [Hz]', fontsize=16)
            # ax1.tick_params(axis='y', labelsize=14)
            # ax1.legend(loc='upper left', fontsize=14)
        
            # # Power Values Plot (Right Y-axis)
            # ax2 = ax1.twinx()
            # ax2.plot(t, deltap, color='#9467bd', linestyle='-', label='∆Power')
            # ax2.plot(t, contingency, color='#d62728', linestyle='-', label='Contingency')
            # ax2.plot(t, primaryreserve, color='#2ca02c', linestyle='-', label='PrimaryReserve')
            # ax2.plot(t, ffr, color='#17becf', linestyle='-', label='FFR')
            # ax2.set_ylabel('Power [MW]', fontsize=16)
            # ax2.tick_params(axis='y', labelsize=14)
            # ax2.legend(loc='upper right', fontsize=14)
            
            # # Sincronizar ambos ejes Y para que coincidan los ceros
            # ax1.set_ylim(-1, 1)  # Ajusta los límites según tu gráfica
            # ax2.set_ylim(-500, 500)   # Asegura la misma escala proporcional entre ambos

            # # Configuración de cuadrícula en ambos ejes
            # ax1.grid(True, which='both', linestyle='--', linewidth=0.7, color='gray', alpha=0.7)
            
            # # Configuración de locators (frecuencia de marcas)
            # ax1.xaxis.set_major_locator(plt.MultipleLocator(5))  # Cada 5 unidades en el eje x
            # ax1.yaxis.set_major_locator(plt.MultipleLocator(0.1))  # Cada 0.1 en el eje y
           
            # # Overall Plot Settings
            # plt.title(f'Power Swing Contingency {contingency_count}', fontsize=18)
            # plt.xlim(0, 60)
            # plt.tight_layout()
            # plt.show()
           ####################################################################### 
            # # PLOT IN P.U.
            # # Plotting code remains the same
            # fig, ax1 = plt.subplots(figsize=(12, 8))
        
            # # # Frequency Values Plot (Left Y-axis)
            # ax1.plot(t, f/50, color='tab:blue', linestyle='-', label='$∆f_{COI}$')
            # ax1.plot(t, der_f/50, color='tab:orange', linestyle='-', label='$d(∆f_{COI})$')
            # ax1.set_xlabel('Time [s]', fontsize=16)
            # ax1.set_ylabel('Frequency [p.u.]', color='tab:blue', fontsize=16)
            # ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=14)
            # ax1.legend(loc='upper left', fontsize=14)
        
            # # Power Values Plot (Right Y-axis)
            # ax2 = ax1.twinx()
            # ax2.plot(t, deltap/row['TotalOutputPower[MW]'], color='tab:green', linestyle='--', label='∆Power')
            # ax2.plot(t, contingency/row['TotalOutputPower[MW]'], color='tab:red', linestyle='--', label='Contingency')
            # ax2.plot(t, primaryreserve/row['TotalOutputPower[MW]'], color='tab:purple', linestyle='--', label='PrimaryReserve')
            # ax2.plot(t, ffr/row['TotalOutputPower[MW]'], color='tab:orange', linestyle='--', label='FFR')
            # ax2.set_ylabel('Power [p.u.]', color='tab:red', fontsize=16)
            # ax2.tick_params(axis='y', labelcolor='tab:red', labelsize=14)
            # ax2.legend(loc='upper right', fontsize=14)
        
            # # Overall Plot Settings
            # plt.title('Power Swing', fontsize=18)
            # plt.xlim(0, 60)
            # plt.tight_layout()
            # plt.show()
            
        else:
            print(f"Contingency {counter}")
            print("No solution found within the specified range of H and k that satisfies constraints.")
            
            # Add the values of k and H to the DataFrame
            results_df['min_K1[GW/Hz]'] = 0
            results_df['min_K[GW/Hz]'] = 0
            results_df['min_H[GW*s]'] = 0
            
            # Store the results_df in the dictionary
            freq_security[f"Contingency{contingency_count}"] = results_df    
            
            # Guardar el DataFrame en una pestaña numerada
            sheet_name = f'Contingency{contingency_count}'
            results_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            contingency_count += 1  # Incrementar el contador de hojas
            

        counter += 1
# Crear un DataFrame vacío para almacenar los resultados
summary = pd.DataFrame(columns=['TIMESTAMP', 'Contingency[MW]', 'Inertia[GW*s]','PrimaryReserve[MW]', 'Gain_PFR[MW/Hz]','FFR[MW]', 'Gain_FFR[MW/Hz]'])


# Iterar sobre los DataFrames en el diccionario y tomar el primer valor de 'A' y 'B'
for nombre_df, df in freq_security.items():
    primer_timestamp = df['TIMESTAMP'].iloc[0]
    primer_contingency = df['Contingency[MW]'].iloc[10]  # Primer valor de la columna 'Contingency[MW]'
    primer_inertia = df['min_H[GW*s]'].iloc[0]  # Primer valor de la columna 'min_H[GW*s'
    # max_primaryreserve = df['PrimaryReserve[MW]'].max()  # Valor maximo de la columna 'PrimaryResponse'
    max_primaryreserve = df['PrimaryReserve[MW]'].iloc[609]  # Valor en el que se estabiliza 'PrimaryReserve[MW]'
    # max_ffr = df['FFR[MW]'].max()  # Valor maximo de la columna 'FFR'
    max_ffr = df['FFR[MW]'].iloc[609]  # Valor en el que se estabiliza 'FFR[MW]'
    primer_gain = df['min_K[GW/Hz]'].iloc[0]  # Primer valor de la columna 'min_K[GW/Hz]'
    primer_gain1 = df['min_K1[GW/Hz]'].iloc[0]  # Primer valor de la columna 'min_K[GW/Hz]'


    # Crear un nuevo DataFrame con los valores actuales
    new_row = pd.DataFrame({'TIMESTAMP': primer_timestamp, 'Contingency[MW]': primer_contingency,
                            'Inertia[GW*s]': primer_inertia, 'PrimaryReserve[MW]': max_primaryreserve, 'Gain_PFR[MW/Hz]': primer_gain, 'FFR[MW]': max_ffr, 'Gain_FFR[MW/Hz]': primer_gain1}, index=[0])

    # Concatenar el nuevo DataFrame a 'summary'
    summary = pd.concat([summary, new_row], ignore_index=True)

Contingency.set_index('TIMESTAMP', inplace=True)
summary.set_index('TIMESTAMP', inplace=True)


#%%
# Save the results of the frequency security constraints analisys
summary.to_csv('../Simulations/BOLIVIA_2050_NZE_FFR/Summary.csv', index=False)
Contingency.to_csv('../Simulations/BOLIVIA_2050_NZE_FFR/Contingency.csv', index=False)
with pd.ExcelWriter('../Simulations/BOLIVIA_2050_NZE_FFR/frec_sec_const.xlsx', engine='xlsxwriter') as writer:
    for contingency, df in freq_security.items():
        df.to_excel(writer, sheet_name=contingency, index=False)

#%% ########RESULTADOS EN CSV

# # # results['CapacityMargin'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/CapacityMargin.csv', header=True, index=True)
# # # #results['EESH-ExpectedEnergyNotServed'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/EESH-ExpectedEnergyNotServed.csv', header=True, index=True)
# # # results['ENSH-EnergyNotServedHourly'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/ENSH-EnergyNotServedHourly.csv', header=True, index=True)
# # # results['ENSR-EnergyNotServedRamping'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/ENSR-EnergyNotServedRamping.csv', header=True, index=True)
# # # results['H2ShadowPrice'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/H2ShadowPrice.csv', header=True, index=True)
# # # results['HeatShadowPrice'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/HeatShadowPrice.csv', header=True, index=True)
# # # # results['LOLF-LosOfLoadFrequency'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/LOLF-LosOfLoadFrequency.csv', header=True, index=True)
# # # # results['LOLH-LosOfLoadHours'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/LOLH-LosOfLoadHours.csv', header=True, index=True)
# # # # results['LOLP-LosOfLoadProbability'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/LOLP-LosOfLoadProbabilityy.csv', header=True, index=True)
# # # results['LostLoad_2D'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/LostLoad_2D.csv', header=True, index=True)
# # # results['LostLoad_2U'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/LostLoad_2U.csv', header=True, index=True)
# # # results['LostLoad_3U'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/LostLoad_3U.csv', header=True, index=True)
# # # results['LostLoad_MaxPower'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/LostLoad_MaxPower.csv', header=True, index=True)
# # # results['LostLoad_MinPower'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/LostLoad_MinPower.csv', header=True, index=True)
# # # results['LostLoad_RampDown'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/LostLoad_RampDown.csv', header=True, index=True)
# # # results['LostLoad_RampUp'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/LostLoad_RampUp.csv', header=True, index=True)
# # # # results['LostLoad_WaterSlack'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/LostLoad_WaterSlack.csv', header=True, index=True)
# # # results['NodalPowerConsumption'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/NodalPowerConsumption.csv', header=True, index=True)
# # # results['OutputCommitted'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/OutputCommitted.csv', header=True, index=True)
# # # results['OutputCostRampUpH'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/OutputCostRampUpH.csv', header=True, index=True)
# # # results['OutputCostStartUpH'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/OutputCostStartUpH.csv', header=True, index=True)
# # # results['OutputCurtailedHeat'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/OutputCurtailedHeat.csv', header=True, index=True)
# # # results['OutputCurtailedPower'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/OutputCurtailedPower.csv', header=True, index=True)
# # # results['OutputCurtailmentReserve_2U'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/OutputCurtailmentReserve_2U.csv', header=True, index=True)
# # # results['OutputCurtailmentReserve_3U'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/OutputCurtailmentReserve_3U.csv', header=True, index=True)
# # # results['OutputDemand_2D'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/OutputDemand_2D.csv', header=True, index=True)
# # # results['OutputDemand_2U'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/OutputDemand_2U.csv', header=True, index=True)
# # # results['OutputDemand_3U'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/OutputDemand_3U.csv', header=True, index=True)
# # # results['OutputDemandModulation'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/OutputDemandModulation.csv', header=True, index=True)
# # # results['OutputEmissions'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/OutputEmissions.csv', header=True, index=True)
# # # results['OutputFlow'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/OutputFlow.csv', header=True, index=True)
# # # results['OutputH2Output'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/OutputH2Output.csv', header=True, index=True)
# # # results['OutputH2Slack'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/OutputH2Slack.csv', header=True, index=True)
# # # results['OutputHeat'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/OutputHeat.csv', header=True, index=True)
# # # results['OutputHeatSlack'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/OutputHeatSlack.csv', header=True, index=True)
# # # results['OutputMaxOutageDown'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/OutputMaxOutageDown.csv', header=True, index=True)
# # # results['OutputMaxOutageUp'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/OutputMaxOutageUp.csv', header=True, index=True)
# # # # results['OutputOptimalityGap'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/OutputOptimalityGap.csv', header=True, index=True)
# # # # results['OutputOptimizationCheck'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/OutputOptimizationCheck.csv', header=True, index=True)
# # # # results['OutputOptimizationError'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/OutputOptimizationError.csv', header=True, index=True)
# # # results['OutputPower'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/OutputPower.csv', header=True, index=True)
# # # results['OutputPowerConsumption'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/OutputPowerConsumption.csv', header=True, index=True)
# # # results['OutputPowerMustRun'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/OutputPowerMustRun.csv', header=True, index=True)
# # # results['OutputPtLDemand'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/OutputPtLDemand.csv', header=True, index=True)
# # # results['OutputRampRate'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/OutputRampRate.csv', header=True, index=True)
# # # results['OutputReserve_2D'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/OutputReserve_2D.csv', header=True, index=True)
# # # results['OutputReserve_2U'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/OutputReserve_2U.csv', header=True, index=True)
# # # results['OutputReserve_3U'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/OutputReserve_3U.csv', header=True, index=True)
# # # results['OutputShedLoad'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/OutputShedLoad.csv', header=True, index=True)
# # # results['OutputShutDown'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/OutputShutDown.csv', header=True, index=True)
# # # results['OutputSpillage'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/OutputSpillage.csv', header=True, index=True)
# # # results['OutputStartUp'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/OutputStartUp.csv', header=True, index=True)
# # # results['OutputStorageInput'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/OutputStorageInput.csv', header=True, index=True)
# # # results['OutputStorageLevel'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/OutputStorageLevel.csv', header=True, index=True)
# # # results['OutputStorageSlack'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/OutputStorageSlack.csv', header=True, index=True)
# # # # results['OutputSystemCost'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/OutputSystemCost.csv', header=True, index=True)
# # # # results['OutputSystemCostD'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/OutputSystemCostD.csv', header=True, index=True)
# # # results['ShadowPrice'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/ShadowPrice.csv', header=True, index=True)
# # # results['ShadowPrice_2D'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/ShadowPrice_2D.csv', header=True, index=True)
# # # results['ShadowPrice_2U'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/ShadowPrice_2U.csv', header=True, index=True)
# # # results['ShadowPrice_3U'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/ShadowPrice_3U.csv', header=True, index=True)
# # # results['ShadowPrice_RampDown_TC'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/ShadowPrice_RampDown_TC.csv', header=True, index=True)
# # # results['ShadowPrice_RampUp_TC'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/ShadowPrice_RampUp_TC.csv', header=True, index=True)
# # # # results['SMML-SystemMinusesMaximalLoad'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/SMML-SystemMinusesMaximalLoad.csv', header=True, index=True)
# # # # results['SMNL-SystemMinusesNominalLoad'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/SMNL-SystemMinusesNominalLoad.csv', header=True, index=True)
# # # results['status'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/status.csv', header=True, index=True)
# # # results['StorageShadowPrice'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/StorageShadowPrice.csv', header=True, index=True)
# # # results['TotalDemand'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/TotalDemand.csv', header=True, index=True)
# # # results['UnitHourly2DRevenue'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/UnitHourly2DRevenue.csv', header=True, index=True)
# # # results['UnitHourly2URevenue'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/UnitHourly2URevenue.csv', header=True, index=True)
# # # results['UnitHourly3URevenue'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/UnitHourly3URevenue.csv', header=True, index=True)
# # # results['UnitHourlyPowerRevenue'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/UnitHourlyPowerRevenue.csv', header=True, index=True)
# # # results['UnitHourlyProductionCost'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/UnitHourlyProductionCost.csv', header=True, index=True)
# # # results['UnitHourlyProfit'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/UnitHourlyProfit.csv', header=True, index=True)
# # # results['UnitHourlyRampingCost'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/UnitHourlyRampingCost.csv', header=True, index=True)
# # # results['UnitHourlyRevenue'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/UnitHourlyRevenue.csv', header=True, index=True)
# # # results['UnitHourlyStartUpCost'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/UnitHourlyStartUpCost.csv', header=True, index=True)
# # # results['UnitHourlyVariableCost'].to_csv('../Results/FINAL_AREAS_NOCLUSTERING/UnitHourlyVariableCost.csv', header=True, index=True)

# # #%% CONSTRUYENDO TABLAS

# # #TABLA 1
# #%%
# tabla1 = pd.DataFrame(columns = ['ZONA', 'GENERACION HIDRO [TWh]', 'EMISIONES CO2 HIDRO [MT]', 'GENERACION TERMO [TWh]', 'EMISIONES CO2 TERMO [MT]','GENERACION VRES [TWh]', 'EMISIONES CO2 VRES [MT]', 'SPILLAGE [MWh]'], dtype=float )
# hidro = r['UnitData'][(r['UnitData'].Fuel) == 'WAT']
# hidroce = hidro[(hidro.Zone) == 'CE']
# hidrono = hidro[(hidro.Zone) == 'NO']
# hidroor = hidro[(hidro.Zone) == 'OR']
# hidrosu = hidro[(hidro.Zone) == 'SU']

# gas = r['UnitData'][(r['UnitData'].Fuel) == 'GAS']
# oil = r['UnitData'][(r['UnitData'].Fuel) == 'OIL']
# win = r['UnitData'][(r['UnitData'].Fuel) == 'WIN']
# sun = r['UnitData'][(r['UnitData'].Fuel) == 'SUN']
# bio = r['UnitData'][(r['UnitData'].Fuel) == 'BIO']
# termo = pd.concat([gas, oil, bio], axis=0)
# termoce = termo[(termo.Zone) == 'CE']
# termono = termo[(termo.Zone) == 'NO']
# termoor = termo[(termo.Zone) == 'OR']
# termosu = termo[(termo.Zone) == 'SU']

# win = pd.concat([win], axis=0)
# wince = win[(win.Zone) == 'CE']
# winno = win[(win.Zone) == 'NO']
# winor = win[(win.Zone) == 'OR']
# winsu = win[(win.Zone) == 'SU']

# sun = pd.concat([sun], axis=0)
# sunce = sun[(sun.Zone) == 'CE']
# sunno = sun[(sun.Zone) == 'NO']
# sunor = sun[(sun.Zone) == 'OR']
# sunsu = sun[(sun.Zone) == 'SU']

# ghidroce  = hidroce['Generation [TWh]'].sum()
# ghidrono  = hidrono['Generation [TWh]'].sum()
# ghidroor  = hidroor['Generation [TWh]'].sum()
# ghidrosu  = hidrosu['Generation [TWh]'].sum()

# gtermoce  = termoce['Generation [TWh]'].sum()
# gtermono  = termono['Generation [TWh]'].sum()
# gtermoor  = termoor['Generation [TWh]'].sum()
# gtermosu  = termosu['Generation [TWh]'].sum()

# gwince  = wince['Generation [TWh]'].sum()
# gwinno  = winno['Generation [TWh]'].sum()
# gwinor  = winor['Generation [TWh]'].sum()
# gwinsu  = winsu['Generation [TWh]'].sum()

# gsunce  = sunce['Generation [TWh]'].sum()
# gsunno  = sunno['Generation [TWh]'].sum()
# gsunor  = sunor['Generation [TWh]'].sum()
# gsunsu  = sunsu['Generation [TWh]'].sum()

# ehidroce  = hidroce['CO2 [t]'].sum()
# ehidrono  = hidrono['CO2 [t]'].sum()
# ehidroor  = hidroor['CO2 [t]'].sum()
# ehidrosu  = hidrosu['CO2 [t]'].sum()

# etermoce  = termoce['CO2 [t]'].sum()
# etermono  = termono['CO2 [t]'].sum()
# etermoor  = termoor['CO2 [t]'].sum()
# etermosu  = termosu['CO2 [t]'].sum()

# ewince  = wince['CO2 [t]'].sum()
# ewinno  = winno['CO2 [t]'].sum()
# ewinor  = winor['CO2 [t]'].sum()
# ewinsu  = winsu['CO2 [t]'].sum()

# esunce  = sunce['CO2 [t]'].sum()
# esunno  = sunno['CO2 [t]'].sum()
# esunor  = sunor['CO2 [t]'].sum()
# esunsu  = sunsu['CO2 [t]'].sum()

# etotce  = wince['CO2 [t]'].sum()
# etotno  = winno['CO2 [t]'].sum()
# etotor  = winor['CO2 [t]'].sum()
# etotsu  = winsu['CO2 [t]'].sum()

# # Selecciona las filas correspondientes a la zona central en 'hidroce'
# centrales_zona_central = hidroce.index

# # Verifica qué centrales en 'centrales_zona_central' están presentes en 'results['OutputSpillage']'
# centrales_presentes = [central for central in centrales_zona_central if central in results['OutputSpillage'].columns]

# # Si hay centrales presentes, realiza la suma
# if centrales_presentes:
#     # Filtra el DataFrame 'results['OutputSpillage']' solo para las centrales presentes
#     spillage_zona_central_subset = results['OutputSpillage'][centrales_presentes]
    
#     # Suma las columnas correspondientes en 'spillage_zona_central_subset'
#     suma_spillage_zona_central = spillage_zona_central_subset.sum(axis=1)
    
#     # Agrega la serie de tiempo resultante como una nueva columna en 'results'
#     results['SumaSpillageZonaCentral'] = suma_spillage_zona_central

#     # Suma total de la columna 'SumaSpillageZonaCentral'
#     spi_ce = results['SumaSpillageZonaCentral'].sum()


# else:
#     print("No hay centrales de la zona central presentes en 'results['OutputSpillage']'")

# # Selecciona las filas correspondientes a la zona central en 'hidroce'
# centrales_zona_norte = hidrono.index

# # Verifica qué centrales en 'centrales_zona_central' están presentes en 'results['OutputSpillage']'
# centrales_presentes = [central for central in centrales_zona_norte if central in results['OutputSpillage'].columns]

# # Si hay centrales presentes, realiza la suma
# if centrales_presentes:
#     # Filtra el DataFrame 'results['OutputSpillage']' solo para las centrales presentes
#     spillage_zona_norte_subset = results['OutputSpillage'][centrales_presentes]
    
#     # Suma las columnas correspondientes en 'spillage_zona_central_subset'
#     suma_spillage_zona_norte = spillage_zona_norte_subset.sum(axis=1)
    
#     # Agrega la serie de tiempo resultante como una nueva columna en 'results'
#     results['SumaSpillageZonaNorte'] = suma_spillage_zona_norte

#     # Suma total de la columna 'SumaSpillageZonaCentral'
#     spi_no = results['SumaSpillageZonaNorte'].sum()


# else:
#     print("No hay centrales de la zona norte presentes en 'results['OutputSpillage']'")
# # Selecciona las filas correspondientes a la zona central en 'hidroce'
# centrales_zona_oriental = hidroor.index

# # Verifica qué centrales en 'centrales_zona_central' están presentes en 'results['OutputSpillage']'
# centrales_presentes = [central for central in centrales_zona_oriental if central in results['OutputSpillage'].columns]

# # Si hay centrales presentes, realiza la suma
# if centrales_presentes:
#     # Filtra el DataFrame 'results['OutputSpillage']' solo para las centrales presentes
#     spillage_zona_oriental_subset = results['OutputSpillage'][centrales_presentes]
    
#     # Suma las columnas correspondientes en 'spillage_zona_central_subset'
#     suma_spillage_zona_oriental = spillage_zona_oriental_subset.sum(axis=1)
    
#     # Agrega la serie de tiempo resultante como una nueva columna en 'results'
#     results['SumaSpillageZonaOriental'] = suma_spillage_zona_oriental

#     # Suma total de la columna 'SumaSpillageZonaCentral'
#     spi_or = results['SumaSpillageZonaOriental'].sum()
#     spi_or = 0

# else:
#     print("No hay centrales de la zona oriental presentes en 'results['OutputSpillage']'")
# # Selecciona las filas correspondientes a la zona central en 'hidroce'
# centrales_zona_sud = hidrosu.index

# # Verifica qué centrales en 'centrales_zona_central' están presentes en 'results['OutputSpillage']'
# centrales_presentes = [central for central in centrales_zona_sud if central in results['OutputSpillage'].columns]

# # Si hay centrales presentes, realiza la suma
# if centrales_presentes:
#     # Filtra el DataFrame 'results['OutputSpillage']' solo para las centrales presentes
#     spillage_zona_sud_subset = results['OutputSpillage'][centrales_presentes]
    
#     # Suma las columnas correspondientes en 'spillage_zona_central_subset'
#     suma_spillage_zona_sud = spillage_zona_sud_subset.sum(axis=1)
    
#     # Agrega la serie de tiempo resultante como una nueva columna en 'results'
#     results['SumaSpillageZonaSud'] = suma_spillage_zona_sud

#     # Suma total de la columna 'SumaSpillageZonaCentral'
#     spi_su = results['SumaSpillageZonaSud'].sum()


# else:
#     print("No hay centrales de la zona sud presentes en 'results['OutputSpillage']'")


# datat1 = data=[['CE',ghidroce,ehidroce,gtermoce,etermoce,gwince,ewince,gsunce,esunce,spi_ce],
#                 ['NO',ghidrono,ehidrono,gtermono,etermono,gwinno,ewinno,gsunno,esunno,spi_no],
#                 ['OR',ghidroor,ehidroor,gtermoor,etermoor,gwinor,ewinor,gsunor,esunor,spi_or],
#                 ['SU',ghidrosu,ehidrosu,gtermosu,etermosu,gwinsu,ewinsu,gsunsu,esunsu,spi_su]]

# tabla1=pd.DataFrame(datat1,columns = ['Zona','Generacion HIDRO [TWh]', 'Emisiones CO2 HIDRO [MT]', 'Generacion TERMO [TWh]', 'Emisiones CO2 TERMO [MT]','Generacion WIN [TWh]', 'Emisiones CO2 WIN [MT]','Generacion SUN [TWh]', 'Emisiones CO2 SUN [MT]', 'Spillage [MWh]'])


# Demand_2D = results['OutputDemand_2D'].sum()
# Demand_3U = results['OutputDemand_3U'].sum()
# NegOutputFlow = results['OutputFlow'][results['OutputFlow'] < 0].sum()
# PosOutputFlow = results['OutputFlow'][results['OutputFlow'] > 0].sum()
# AbsOutputFlow = results['OutputFlow'].sum()


# #%%
# # Selecciona las filas correspondientes a la zona central en 'hidroce'
# ycentrales_hidrono = hidrono.index

# # Verifica qué centrales en 'centrales_zona_central' están presentes en 'results['OutputSpillage']'
# centrales_presentes = [central for central in centrales_hidrono if central in results['OutputPower'].columns]

# # Definición del intervalo de fechas
# start = '2030-01-01 00:00:00+00:00'
# end = '2030-06-30 23:00:00+00:00'
# start = pd.to_datetime(start)
# end = pd.to_datetime(end)

# # Filtrado del DataFrame por el intervalo de fechas
# df = results['OutputPower'].loc[start:end]

# # Si hay centrales presentes, realiza la suma
# if centrales_presentes:
#     # Filtra el DataFrame 'results['OutputSpillage']' solo para las centrales presentes
#     power_hidrono_subset = df[centrales_presentes]
    
#     # Suma las columnas correspondientes en 'spillage_zona_central_subset'
#     suma_power_hidrono = power_hidrono_subset.sum(axis=1)
    
#     # Agrega la serie de tiempo resultante como una nueva columna en 'results'
#     results['SumaPowerhidrono'] = suma_power_hidrono

#     # Suma total de la columna 'SumaSpillageZonaCentral'
#     pow_hidrono = results['SumaPowerhidrono'].sum()


# else:
#     print("No hay centrales de la zona central presentes en 'results['OutputPower']'")
# #%%
# # ### SHEDLOAD por zona

# # outshed = results['OutputShedLoad'].sum()
# # outshed['OR'] = 0 
# # outshed['SU'] = 0 
# # outshed = outshed.to_frame(name='OutputShedLoad')


# # ### CURTAILMENT por zona

# # outcurt = results['OutputCurtailedPower'].sum()
# # outcurt['NO'] = 0 
# # outcurt = outcurt.to_frame(name='OutputCurtailedPower')
# # outcurt.reset_index(level =0, inplace = True)
# # outcurt['newindex'] = [0,2,3,1]
# # outcurt.set_index('newindex',inplace=True, drop=True)	
# # outcurt = outcurt.sort_index()
# # outcurt.set_index('index',inplace=True, drop=True)	

# # ### LOSTLOADS por zona

# # outlost = pd.DataFrame()
# # outlost['Zona'] = ['CE','NO','OR','SU'] 
# # outlost['LostLoads'] = [0,0,0,0] 
# # outlost.set_index('Zona',inplace=True, drop=True)	

# # ### SHADOWPRICES por zona

# # outshad = results['ShadowPrice'].mean()
# # outshad = outshad.to_frame(name='ShadowPrice')

# # ###Spillage por zona
# # import numpy as np
# # outspi = results['OutputSpillage']
# # outspi1= outspi.transpose()
# # outspi1.reset_index(level =0, inplace = True)

# # buslist = r['UnitData']
# # # buslist = buslist.drop(['level_0'], axis=1)
# # buslist.reset_index(level =0, inplace = True)
# # buslist =  buslist[['index','Zone']]
# # buslist = buslist.rename({'index':'Busname'}, axis=1)	

# # busname = buslist.Busname
# # busname.to_dict()
# # busname = np.asarray(busname)
# # zone = buslist.Zone
# # zone.to_dict()
# # zone = np.asarray(zone)
# # for i, j in zip(busname, zone):
# #     outspi1['index'] = outspi1['index'].str.replace(i,j, regex=False)

# # outspi2= outspi1.transpose()
# # outspi2.columns = outspi2.iloc[0]
# # outspi2 = outspi2[1:]
# # outspi3 = outspi2.groupby(level=0, axis=1).sum()

# # outspice  = outspi3['CE'].sum()
# # outspino  = outspi3['NO'].sum()
# # outspisu  = outspi3['SU'].sum()
# # outspior  = 0

# # datat2 = data2=[['CE',ghidroce,ehidroce,gtermoce,etermoce,gvresce,evresce,outspice],
# #                ['NO',ghidrono,ehidrono,gtermono,etermono,gvresno,evresno,outspino],
# #                ['OR',ghidroor,ehidroor,gtermoor,etermoor,gvresor,evresor,outspior],
# #                ['SU',ghidrosu,ehidrosu,gtermosu,etermosu,gvressu,evressu,outspisu]]

# # tabla2=pd.DataFrame(datat2,columns = ['Zona','Generacion HIDRO [TWh]', 'Emisiones CO2 HIDRO [MT]', 
# #                                       'Generacion TERMO [TWh]', 'Emisiones CO2 TERMO [MT]','Generacion VRES [TWh]', 
# #                                       'Emisiones CO2 VRES [MT]','OutputSpillage'])
# # tabla2.set_index('Zona',inplace=True, drop=True)
# # frames = [tabla2, outshed, outcurt, outlost, outshad]
# # tabla2 = pd.concat(frames,axis=1)

# # tabla2.to_csv('../Results/ENDE_SCENARIO1/1.resumen.csv', header=True, index=True)


# # #%% PLOT GENERATION BY ZONES

# # import numpy as np
# # outpow = results['OutputPower']
# # outpow1= outpow.transpose()
# # outpow1.reset_index(inplace = True)

# # buslist = r['UnitData']
# # # buslist = buslist.drop(['level_0'], axis=1)
# # # buslist.reset_index(inplace = True)
# # buslist =  buslist[['index','Fuel','Zone']]
# # buslist['FuelZone'] = buslist['Fuel'] + '-' + buslist['Zone']
# # buslist = buslist.rename({'index':'Busname'}, axis=1)	

# # busname = buslist.Busname
# # busname.to_dict()
# # busname = np.asarray(busname)
# # fuelzone = buslist.FuelZone
# # fuelzone.to_dict()
# # fuelzone = np.asarray(fuelzone)
# # for i, j in zip(busname, fuelzone):
# #     outpow1['index'] = outpow1['index'].str.replace(i,j, regex=False)

# # outpow2= outpow1.transpose()
# # outpow2.columns = outpow2.iloc[0]
# # outpow2 = outpow2[1:]
# # outpow3 = outpow2.groupby(level=0, axis=1).sum()

# # df = pd.DataFrame()


# # df = outpow3
# # df['Datetime'] = pd.date_range(start='2025-12-31 23:00:00+00:00', end='2026-12-31 22:00:00+00:00', freq='H')
# # df['Datetime'] = pd.to_datetime(df['Datetime'])
# # df = df.drop(['level_0','index'], axis=1)
# # df = df.rename({'Datetime':'TIMESTAMP'}, axis=1)	


# # df['Total Hidro'] = df['WAT-CE'] + df['WAT-NO'] + df['WAT-SU']
# # df['Total Term'] = df['GAS-CE'] + df['GAS-OR'] + df['GAS-NO'] + df['GAS-SU'] + df['BIO-NO'] + df['BIO-OR']
# # df['Total Renov'] = df['SUN-CE'] + df['SUN-SU'] + df['WIN-CE'] + df['WIN-OR'] + df['WIN-SU']


# # mask1 = (df['TIMESTAMP'] > '2026-06-01 22:00:00+00:00') & (df['TIMESTAMP'] <= '2026-06-07 23:00:00+00:00')
# # df = df.loc[mask1]

# # #%% STACKED AREA CHART
# # import numpy as np
# # import pandas as pd
# # from plotly.offline import plot
# # import plotly.graph_objs as go


# # #CENTRAL
# # x = df['TIMESTAMP']

# # gasce = dict(
# #     x = x,
# #     y = df['GAS-CE'],
# #     hovertemplate = '<b>%{x}'+
# #                    '<br>Gas CE: <b>%{y}',
# #     mode = 'lines',
# #     line = dict(width = 0.5,
# #                 color = 'darkorange'),
# #     stackgroup = 'one'
# #     )

# # watce = dict(
# #     x = x,
# #     y = df['WAT-CE'],
# #     hovertemplate = '<b>%{x}'+
# #                 '<br>Hidro CE: <b>%{y}',
# #     mode = 'lines',
# #     line = dict(width = 0.5,
# #                 color = 'steelblue'),
# #     stackgroup = 'one'
# #     )

# # wince = dict(
# #     x = x,
# #     y = df['WIN-CE'],
# #     hovertemplate = '<b>%{x}'+
# #                 '<br>Wind CE: <b>%{y}',
# #     mode = 'lines',
# #     line = dict(width = 0.5,
# #                 color = 'forestgreen'),
# #     stackgroup = 'one'
# #     )

# # sunce = dict(
# #     x = x,
# #     y = df['SUN-CE'],
# #     hovertemplate = '<b>%{x}'+
# #                 '<br>Sun CE: <b>%{y}',
# #     mode = 'lines',
# #     line = dict(width = 0.5,
# #                 color = 'yellow'),
# #     stackgroup = 'one'
# #     )

# # data = [gasce, watce, wince, sunce]

# # fig = dict(data = data)
# # hovermode = "x unified"

# # plot(fig, filename = 'stacked-area-plot-hover', validate = False)





# # # #%%
# # # #filtrar valores no nulos de OutputSpillage
# # # OutputSpillage = results['OutputSpillage']
# # # OutputSpillage_mask = OutputSpillage.sum(axis = 1) != 0 
# # # OutputSpillage_filtered = OutputSpillage[OutputSpillage_mask]  #ver esto
# # # OutputSpillage_filtered.reset_index(level =0, inplace = True)
# # # values = OutputSpillage_filtered['index']
# # # OutputSpillage_filtered.to_csv('../Results/FINAL_AREAS_NOCLUSTERING/2.OutputSpillage_filtered.csv', header=True, index=True)

# # # #filtrar fechas con spillage en OutputFlow
# # # OutputFlow = results['OutputFlow']
# # # OutputFlow_filtered = OutputFlow[OutputFlow['index'].isin(values)] #ver esto
# # # OutputFlow_filtered.to_csv('../Results/FINAL_AREAS_NOCLUSTERING/3.OutputFlow_filtered.csv', header=True, index=True)

# # # #filtrar fechas con spillage en OutputPower
# # # units_spillage = OutputSpillage.columns
# # # OutputPower = results['OutputPower']
# # # OutputPower = OutputPower[units_spillage]
# # # OutputPower.reset_index(level =0, inplace = True)
# # # OutputPower_filtered = OutputPower[OutputFlow['index'].isin(values)] #ver esto
# # # OutputPower_filtered['index'] = OutputFlow_filtered['index']
# # # OutputPower_filtered.to_csv('../Results/FINAL_AREAS_NOCLUSTERING/4.OutputPower_filtered.csv', header=True, index=True)

# # # #promedio de shadowprice
# # # ShadowPrice = results['ShadowPrice']
# # # ShadowPriceCE = ShadowPrice['CE'].mean()
# # # ShadowPriceNO = ShadowPrice['NO'].mean()
# # # ShadowPriceOR = ShadowPrice['OR'].mean()
# # # ShadowPriceSU = ShadowPrice['SU'].mean()
# # # #%% OTRAS TABLAS
# # # # TABLA 5 DATOS DEL SISTEMA POR ZONAS

# # # tabla5 = pd.concat([results['CapacityMargin'].sum(), results['EENS-ExpectedEnergyNotServed'],
# # #                     results['ENSH-EnergyNotServedHourly'].sum(),
# # #                     results['ENSR-EnergyNotServedRamping'].sum(),
# # #                     results['LOLF-LosOfLoadFrequency'],
# # #                     results['LOLH-LosOfLoadHours'],
# # #                     results['LOLP-LosOfLoadProbability'],
# # #                     results['LostLoad_2D'].sum(),results['LostLoad_2U'].sum(),
# # #                     results['LostLoad_3U'].sum(),results['LostLoad_MaxPower'].sum(),
# # #                     results['LostLoad_MinPower'].sum(),results['LostLoad_RampDown'].transpose (),
# # #                     results['OutputCurtailmentReserve_2U'].sum(),
# # #                     results['OutputCurtailmentReserve_3U'].sum(),results['OutputDemand_2D'].sum(),
# # #                     results['OutputDemand_2U'].sum(),results['OutputDemand_3U'].sum(),
# # #                     results['OutputMaxOutageDown'].sum(),results['OutputMaxOutageUp'].sum(),
# # #                     results['OutputShedLoad'].sum(),results['SMML-SystemMinusesMaximalLoad'],
# # #                     results['SMNL-SystemMinusesNominalLoad']], axis=1)

# # # tabla5.rename(columns = {0:'CapacityMargin',1:'EENS-ExpectedEnergyNotServed',
# # #                          2:'ENSH-EnergyNotServedHourly',3:'ENSR-EnergyNotServedRamping',
# # #                          4:'LOLF-LosOfLoadFrequency',5:'LOLH-LosOfLoadHours',
# # #                          6:'LOLP-LosOfLoadProbability',7:'LostLoad_2D',8:'LostLoad_2U',
# # #                          9:'LostLoad_3U',10:'LostLoad_MaxPower',11:'LostLoad_MinPower',
# # #                          '2026-01-01 00:00:00':'LostLoad_RampDown',
# # #                          12:'OutputCurtailmentReserve_2U',13:'OutputCurtailmentReserve_3U',
# # #                          14:'OutputDemand_2D',15:'OutputDemand_2U',16:'OutputDemand_3U',
# # #                          17:'OutputMaxOutageDown',18:'OutputMaxOutageUp',19:'OutputShedLoad',
# # #                          20:'SMML-SystemMinusesMaximalLoad',21:'SMNL-SystemMinusesNominalLoad'}, 
# # #               inplace=True)

# # # ########PONER ESTO EN RESERVA ROTANTE Y FRIA
# # # ShadowPrice = pd.DataFrame(columns=['ShadowPrice'])
# # # ShadowPrice['ShadowPrice'] = results['ShadowPrice']

# # # ShadowPrice_2D = pd.DataFrame(columns=['ShadowPrice_2D'])
# # # ShadowPrice_2D['ShadowPrice_2D'] = results['ShadowPrice_2D']

# # # ShadowPrice_2U = pd.DataFrame(columns=['ShadowPrice_2U'])
# # # ShadowPrice_2U['ShadowPrice_2U'] = results['ShadowPrice_2U']

# # # ShadowPrice_3U = pd.DataFrame(columns=['ShadowPrice_3U'])
# # # ShadowPrice_3U['ShadowPrice_3U'] = results['ShadowPrice_3U']

# # # StorageShadowPrice = pd.DataFrame(columns=['StorageShadowPrice'])
# # # StorageShadowPrice['StorageShadowPrice'] = results['StorageShadowPrice']
# # # #########


# # # ########ANALIZAR DONDE PONER ESTO????

# # # OutputStorageLevel = pd.DataFrame(columns=['OutputStorageLevel'])
# # # OutputStorageLevel['OutputStorageLevel'] = results['OutputStorageLevel'].sum()

# # # OutputPower = pd.DataFrame(columns=['OutputPower'])
# # # OutputPower['OutputPower'] = results['OutputPower'].sum()

# # # OutputStorageSlack = pd.DataFrame(columns=['OutputStorageSlack'])
# # # OutputStorageSlack['OutputStorageSlack'] = results['OutputStorageSlack'].sum()

# # # #########



# # # ########PONER ESTO EN EMISIONES
# # # OutputEmissions = pd.DataFrame(columns=['OutputEmissions'])
# # # OutputEmissions['OutputEmissions'] = results['OutputEmissions'].sum()
# # # #########




# # # ########PONER ESTO EN CONGESTION
# # # OutputFlow = pd.DataFrame(columns=['OutputFlow'])
# # # OutputFlow['OutputFlow'] = results['OutputFlow'].sum()
# # # ############


# # # ########PONER ESTO EN POR UNIDADES DE GENERACION
# # # OutputPower = pd.DataFrame(columns=['OutputPower'])
# # # OutputPower['OutputPower'] = results['OutputPower'].sum()

# # # OutputPowerMustRun = pd.DataFrame(columns=['OutputPowerMustRun'])
# # # OutputPowerMustRun['OutputPowerMustRun'] = results['OutputPowerMustRun'].sum()

# # # OutputRampRate = pd.DataFrame(columns=['OutputRampRate'])
# # # OutputRampRate['OutputRampRate'] = results['OutputRampRate'].sum()

# # # OutputReserve_2D = pd.DataFrame(columns=['OutputReserve_2D'])
# # # OutputReserve_2D['OutputReserve_2D'] = results['OutputReserve_2D'].sum()

# # # OutputReserve_2U = pd.DataFrame(columns=['OutputReserve_2U'])
# # # OutputReserve_2U['OutputReserve_2U'] = results['OutputReserve_2U'].sum()

# # # OutputReserve_3U = pd.DataFrame(columns=['OutputReserve_3U'])
# # # OutputReserve_3U['OutputReserve_3U'] = results['OutputReserve_3U'].sum()

# # # OutputShutDown = pd.DataFrame(columns=['OutputShutDown'])
# # # OutputShutDown['OutputShutDown'] = results['OutputShutDown'].sum()

# # # OutputStartUp = pd.DataFrame(columns=['OutputStartUp'])
# # # OutputStartUp['OutputStartUp'] = results['OutputStartUp'].sum()

# # # OutputStorageLevel = pd.DataFrame(columns=['OutputStorageLevel'])
# # # OutputStorageLevel['OutputStorageLevel'] = results['OutputStorageLevel'].sum()

# # # OutputStorageSlack = pd.DataFrame(columns=['OutputStorageSlack'])
# # # OutputStorageSlack['OutputStorageSlack'] = results['OutputStorageSlack'].sum()



# # # ############COSTO DEL SISTEMA

# # # OutputSystemCost = pd.DataFrame(columns=['OutputSystemCost'])
# # # OutputSystemCost['OutputSystemCost'] = results['OutputSystemCost'].sum()



# #%%
# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# %matplotlib inline
# import itertools
# from itertools import chain
# from sklearn.feature_selection import RFE
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import VotingClassifier
# from sklearn.model_selection import GridSearchCV, cross_val_score, learning_curve, train_test_split
# from sklearn.metrics import precision_score, recall_score, confusion_matrix, roc_curve
# import warnings
# py.init_notebook_mode(connected=True)
# import missingno as msno



# from pandas.plotting import scatter_matrix
# #import missingno as msno
# from datetime import datetime


# # pd.options.display.float_format='{:,.4f}'.format
# # ce = pd.read_csv('C:/Users/navia/Dispa-SET_Bolivia.git/Load/CE/2020.csv')
# # ce = ce.rename({'DEMAND':'DEMAND CE'}, axis=1)

# # no = pd.read_csv('C:/Users/navia/Dispa-SET_Bolivia.git/Load/NO/2020.csv')
# # no = no.rename({'DEMAND':'DEMAND NO'}, axis=1)
# # no = no['DEMAND NO']

# # su = pd.read_csv('C:/Users/navia/Dispa-SET_Bolivia.git/Load/NO/2020.csv')
# # su = su.rename({'DEMAND':'DEMAND SU'}, axis=1)
# # su = su['DEMAND SU']

# # oi = pd.read_csv('C:/Users/navia/Dispa-SET_Bolivia.git/Load/OR/2020.csv')
# # oi = oi.rename({'DEMAND':'DEMAND OR'}, axis=1)
# # oi = oi['DEMAND OR']

# import pandas as pd
# import matplotlib.pyplot as plt

# # Lectura del archivo CSV
# df = pd.read_csv('../Demand/Demand_2025.csv')

# # Conversión de la columna TIMESTAMP a tipo datetime y configuración como índice
# df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
# df.set_index('TIMESTAMP', inplace=True)

# # Definición del intervalo de fechas
# start = '2025-11-01 00:00:00+00:00'
# end = '2025-12-31 23:00:00+00:00'
# start = pd.to_datetime(start)
# end = pd.to_datetime(end)

# # Filtrado del DataFrame por el intervalo de fechas
# df = df.loc[start:end]

# # Configuración del orden deseado de las columnas
# column_order = ['CE', 'NO', 'OR', 'SU']

# # Configuración de colores correspondientes al orden de las columnas
# colors = {'CE': 'orange', 'NO': 'yellow', 'OR': 'green', 'SU': 'blue'}

# # Selección de las columnas y generación del gráfico de áreas
# ax = df[column_order].plot.area(color=[colors[col] for col in column_order], alpha=0.5, figsize=(18, 6), subplots=True)

# # Ajuste de la leyenda para que aparezca en la parte superior izquierda
# ax[0].legend(loc='upper left')

# # Ajustes estéticos y visualización del gráfico
# plt.show()

# #%%
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Supongamos que tienes un DataFrame llamado df1 con las series de tiempo de los AF
# # Asegúrate de tener la columna 'TIMESTAMP' configurada como datetime
# # Tu DataFrame df2
# df = pd.read_csv('../RenewablesAF/RenewablesAF_2030.csv')
# # Conversión de la columna TIMESTAMP a tipo datetime y configuración como índice
# df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
# df.set_index('TIMESTAMP', inplace=True)
# columnas_deseadas = ['ORU', 'UYU', 'YUN', 'ORU2', 'SOLCB1', 'SOLCB2', 'SOLVIN', 'SOLSAN']

# # Seleccionar solo las columnas deseadas y actualizar el DataFrame
# df_filtrado = df[columnas_deseadas]

# # Definición del intervalo de fechas
# start = '2030-11-01 00:00:00+00:00'
# end = '2030-12-31 23:00:00+00:00'
# start = pd.to_datetime(start)
# end = pd.to_datetime(end)

# # Filtrado del DataFrame por el intervalo de fechas
# df = df.loc[start:end]

# # Configuración del orden deseado de las columnas
# column_order = ['ORU', 'UYU', 'YUN', 'ORU2', 'SOLCB1', 'SOLCB2', 'SOLVIN', 'SOLSAN']

# # Configuración de colores correspondientes al orden de las columnas
# colors = {'ORU':'yellow', 'UYU':'yellow', 'YUN':'yellow', 'ORU2':'yellow', 'SOLCB1':'yellow', 'SOLCB2':'yellow', 'SOLVIN':'yellow', 'SOLSAN':'yellow'}

# # Selección de las columnas y generación del gráfico de áreas
# ax = df[column_order].plot.area(color=[colors[col] for col in column_order], alpha=1, figsize=(60, 40), subplots=True)

# # Ajuste de la leyenda para que aparezca en la parte superior izquierda en cada subgráfico
# for ax_sub in ax:
#     ax_sub.legend(loc='upper left', fontsize='38')  # Ajusta el tamaño de la fuente de la leyenda para cada subgráfico

# # Ajustes estéticos
# for i, af_column in enumerate(column_order, 1):
#     ax[i - 1].set_xlabel('Fecha', fontsize='32')  # Ajusta el tamaño de la fuente del eje x
#     ax[i - 1].set_ylabel(f'{af_column}', fontsize='32')  # Ajusta el tamaño de la fuente del eje y
#     ax[i - 1].tick_params(axis='both', labelsize='32')  # Ajusta el tamaño de la fuente de los números en los ejes
#     ax[i - 1].xaxis.set_tick_params(which='both', labelsize='32')  # Ajusta el tamaño de la fuente de los días en el eje x

# # Visualización del gráfico
# plt.show()


# #%%

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Supongamos que tienes un DataFrame llamado df1 con las series de tiempo de los AF
# # Asegúrate de tener la columna 'TIMESTAMP' configurada como datetime
# # Tu DataFrame df2
# df = pd.read_csv('../RenewablesAF/RenewablesAF_2030.csv')
# # Conversión de la columna TIMESTAMP a tipo datetime y configuración como índice
# df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
# df.set_index('TIMESTAMP', inplace=True)
# columnas_deseadas = ['ORU', 'UYU', 'YUN', 'ORU2', 'SOLCB1', 'SOLCB2', 'SOLVIN', 'SOLSAN']

# # Seleccionar solo las columnas deseadas y actualizar el DataFrame
# df_filtrado = df[columnas_deseadas]

# # Definición del intervalo de fechas
# start = '2030-01-01 00:00:00+00:00'
# end = '2030-12-31 23:00:00+00:00'
# start = pd.to_datetime(start)
# end = pd.to_datetime(end)

# # Filtrado del DataFrame por el intervalo de fechas
# df = df.loc[start:end]

# # Selección de las columnas
# columnas_deseadas = ['ORU', 'UYU', 'YUN', 'ORU2', 'SOLCB1', 'SOLCB2', 'SOLVIN', 'SOLSAN']

# # Filtrado del DataFrame por el intervalo de fechas
# df_filtrado = df[columnas_deseadas].loc[start:end]

# # Agregar columnas para mes y hora
# df_filtrado['Month'] = df_filtrado.index.month
# df_filtrado['Hour'] = df_filtrado.index.hour

# # Calcular los promedios mensuales por hora para cada columna
# df_promedios = df_filtrado.groupby(['Month', 'Hour'])[columnas_deseadas].mean().reset_index()

# # Asociar los números de los meses con sus nombres correspondientes
# meses_dict = {1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril', 5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto', 9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'}

# # Ajustes estéticos
# sns.set(style="darkgrid", palette="husl")

# # Configuración del tamaño del lienzo
# plt.figure(figsize=(60, 40))

# # Iterar sobre las columnas y crear un gráfico de líneas para cada columna en un subgráfico
# for i, columna in enumerate(columnas_deseadas, 1):
#     plt.subplot(2, 4, i)
#     ax = sns.lineplot(data=df_promedios, x='Hour', y=columna, hue='Month', palette="husl", linewidth=2)
#     plt.title(f'Promedio Mensual por Hora - {columna}', fontsize=46)
#     plt.xlabel('Hora del Día', fontsize=42)
#     plt.ylabel('Promedio Mensual', fontsize=42)
    
#     # Obtener los nombres de los meses para las leyendas
#     nombres_meses = [meses_dict[mes] for mes in sorted(df_promedios['Month'].unique())]
    
#     # Añadir manualmente la leyenda
#     ax.legend(title='Mes', labels=nombres_meses, fontsize=40)

#     plt.xticks(fontsize=40)  # Ajusta el tamaño de la fuente en el eje x
#     plt.yticks(fontsize=40)  # Ajusta el tamaño de la fuente en el eje y

# # Ajustes estéticos adicionales
# plt.tight_layout()
# plt.show()

# #%%
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Supongamos que tienes un DataFrame llamado df1 con las series de tiempo de los AF
# # Asegúrate de tener la columna 'TIMESTAMP' configurada como datetime
# # Tu DataFrame df2
# df = pd.read_csv('../RenewablesAF/RenewablesAF_2030.csv')
# # Conversión de la columna TIMESTAMP a tipo datetime y configuración como índice
# df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
# df.set_index('TIMESTAMP', inplace=True)
# columnas_deseadas = ['QOL', 'QOL2', 'WAR1', 'SJU', 'EDO', 'VEN', 'PESCZ', 'EDO2']

# # Seleccionar solo las columnas deseadas y actualizar el DataFrame
# df_filtrado = df[columnas_deseadas]

# # Definición del intervalo de fechas
# start = '2030-09-01 00:00:00+00:00'
# end = '2030-10-31 23:00:00+00:00'
# start = pd.to_datetime(start)
# end = pd.to_datetime(end)

# # Filtrado del DataFrame por el intervalo de fechas
# df = df.loc[start:end]

# # Configuración del orden deseado de las columnas
# column_order = ['QOL', 'QOL2', 'WAR1', 'SJU', 'EDO', 'VEN', 'PESCZ', 'EDO2']

# # Configuración de colores correspondientes al orden de las columnas
# colors = {'QOL':'green', 'QOL2':'green', 'WAR1':'green', 'SJU':'green', 'EDO':'green', 'VEN':'green', 'PESCZ':'green', 'EDO2':'green'}

# # Selección de las columnas y generación del gráfico de áreas
# ax = df[column_order].plot.area(color=[colors[col] for col in column_order], alpha=1, figsize=(60, 40), subplots=True)

# # Ajuste de la leyenda para que aparezca en la parte superior izquierda en cada subgráfico
# for ax_sub in ax:
#     ax_sub.legend(loc='upper left', fontsize='38')  # Ajusta el tamaño de la fuente de la leyenda para cada subgráfico

# # Ajustes estéticos
# for i, af_column in enumerate(column_order, 1):
#     ax[i - 1].set_xlabel('Fecha', fontsize='32')  # Ajusta el tamaño de la fuente del eje x
#     ax[i - 1].set_ylabel(f'{af_column}', fontsize='32')  # Ajusta el tamaño de la fuente del eje y
#     ax[i - 1].tick_params(axis='both', labelsize='32')  # Ajusta el tamaño de la fuente de los números en los ejes
#     ax[i - 1].xaxis.set_tick_params(which='both', labelsize='32')  # Ajusta el tamaño de la fuente de los días en el eje x

# # Visualización del gráfico
# plt.show()

# #%%

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Supongamos que tienes un DataFrame llamado df1 con las series de tiempo de los AF
# # Asegúrate de tener la columna 'TIMESTAMP' configurada como datetime
# # Tu DataFrame df2
# df = pd.read_csv('../RenewablesAF/RenewablesAF_2030.csv')
# # Conversión de la columna TIMESTAMP a tipo datetime y configuración como índice
# df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
# df.set_index('TIMESTAMP', inplace=True)
# columnas_deseadas = ['QOL', 'QOL2', 'WAR1', 'SJU', 'EDO', 'VEN', 'PESCZ', 'EDO2']

# # Seleccionar solo las columnas deseadas y actualizar el DataFrame
# df_filtrado = df[columnas_deseadas]

# # Definición del intervalo de fechas
# start = '2030-01-01 00:00:00+00:00'
# end = '2030-12-31 23:00:00+00:00'
# start = pd.to_datetime(start)
# end = pd.to_datetime(end)

# # Filtrado del DataFrame por el intervalo de fechas
# df = df.loc[start:end]

# # Selección de las columnas
# columnas_deseadas = ['QOL', 'QOL2', 'WAR1', 'SJU', 'EDO', 'VEN', 'PESCZ', 'EDO2']

# # Filtrado del DataFrame por el intervalo de fechas
# df_filtrado = df[columnas_deseadas].loc[start:end]

# # Agregar columnas para mes y hora
# df_filtrado['Month'] = df_filtrado.index.month
# df_filtrado['Hour'] = df_filtrado.index.hour

# # Calcular los promedios mensuales por hora para cada columna
# df_promedios = df_filtrado.groupby(['Month', 'Hour'])[columnas_deseadas].mean().reset_index()

# # Asociar los números de los meses con sus nombres correspondientes
# meses_dict = {1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril', 5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto', 9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'}

# # Ajustes estéticos
# sns.set(style="darkgrid", palette="husl")

# # Configuración del tamaño del lienzo
# plt.figure(figsize=(60, 40))

# # Iterar sobre las columnas y crear un gráfico de líneas para cada columna en un subgráfico
# for i, columna in enumerate(columnas_deseadas, 1):
#     plt.subplot(2, 4, i)
#     ax = sns.lineplot(data=df_promedios, x='Hour', y=columna, hue='Month', palette="husl", linewidth=2)
#     plt.title(f'Promedio Mensual por Hora - {columna}', fontsize=46)
#     plt.xlabel('Hora del Día', fontsize=42)
#     plt.ylabel('Promedio Mensual', fontsize=42)
    
#     # Obtener los nombres de los meses para las leyendas
#     nombres_meses = [meses_dict[mes] for mes in sorted(df_promedios['Month'].unique())]
    
#     # Añadir manualmente la leyenda
#     ax.legend(title='Mes', labels=nombres_meses, fontsize=40)

#     plt.xticks(fontsize=40)  # Ajusta el tamaño de la fuente en el eje x
#     plt.yticks(fontsize=40)  # Ajusta el tamaño de la fuente en el eje y

# # Ajustes estéticos adicionales
# plt.tight_layout()
# plt.show()
# #%%
# import pandas as pd
# import matplotlib.pyplot as plt

# # Supongamos que tienes un DataFrame llamado df con las series de tiempo de los AF
# # Asegúrate de tener la columna 'TIMESTAMP' configurada como datetime
# # Tu DataFrame df
# df = pd.read_csv('../ScaledInflows/ScaledInflows_2030.csv')
# df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
# df.set_index('TIMESTAMP', inplace=True)

# # Definición del intervalo de fechas
# start = '2030-01-01 00:00:00+00:00'
# end = '2030-12-31 23:00:00+00:00'
# start = pd.to_datetime(start)
# end = pd.to_datetime(end)

# # Filtrado del DataFrame por el intervalo de fechas
# df = df.loc[start:end]

# # Especifica el número de subplots
# num_subplots = 3

# # Calcula la cantidad de columnas en cada subplot
# cols_per_subplot = len(df.columns) // num_subplots

# # Divide las columnas en partes iguales para cada subplot
# columnas_por_subplot = [df.columns[i:i + cols_per_subplot] for i in range(0, len(df.columns), cols_per_subplot)]

# # Itera sobre las partes y crea un plt.show() para cada una
# for columnas_subset in columnas_por_subplot:
#     # Generación del gráfico de áreas en azul
#     ax = df[columnas_subset].plot.area(color='blue', alpha=1, figsize=(60, 40), subplots=True)

#     # Ajuste de la leyenda para que aparezca en la parte superior izquierda en cada subgráfico
#     for ax_sub in ax:
#         ax_sub.legend(loc='upper left', fontsize='38')  # Ajusta el tamaño de la fuente de la leyenda para cada subgráfico

#     # Ajustes estéticos
#     for ax_sub in ax:
#         ax_sub.set_xlabel('Fecha', fontsize='38')  # Ajusta el tamaño de la fuente del eje x
#         ax_sub.set_ylabel('Inflows', fontsize='38')  # Ajusta el tamaño de la fuente del eje y
#         ax_sub.tick_params(axis='both', labelsize='38')  # Ajusta el tamaño de la fuente de los números en los ejes
#         ax_sub.xaxis.set_tick_params(which='both', labelsize='38')  # Ajusta el tamaño de la fuente de los días en el eje x

#     # Visualización del gráfico
#     plt.show()


# #%% PARA ENCONTRAR TRG Y TG
# vreunits = inputs['units']

# fuels=['SUN','WIN']

# vreunits = vreunits[vreunits.Fuel.isin(fuels)]

# vrelist = list(vreunits['Unit'])

# vreoutputpow = results['OutputPower']

# vreoutputpow = vreoutputpow.T

# vreoutputpow = vreoutputpow.reset_index()

# vreoutputpow.rename(columns = {'index':'Unit'}, inplace = True)

# vreoutputpow = vreoutputpow[vreoutputpow.Unit.isin(vrelist)]

# vreoutputpow = vreoutputpow.T

# vreoutputpow.columns = vreoutputpow.iloc[0]

# vreoutputpow = vreoutputpow[1:]

# vreoutputpow['TRG'] = vreoutputpow.sum(axis=1)

# vreoutputpow['TG'] = results['OutputPower'].sum(axis=1)

# #%%

# import pandas as pd
# import numpy as np

# # Supongamos que 'results' es tu DataFrame con las columnas mencionadas

# # Definir los valores máximos y mínimos
# max_values = {'CE -> SU': 444, 'NO -> CE': 598, 'NO -> OR': 140, 'OR -> CE': 1022}
# min_values = {'CE -> SU': -444, 'NO -> CE': -598, 'NO -> OR': -140, 'OR -> CE': -1022}


# # Contar la cantidad de valores dentro del 80% del valor máximo y del valor mínimo
# count_above_80_percent_max = results['OutputFlow'].apply(lambda col: (col >= max_values[col.name] * 0.8).sum())
# count_below_80_percent_min = results['OutputFlow'].apply(lambda col: (col <= min_values[col.name] * 0.8).sum())

# print("Cantidad de valores por encima del 80% del valor máximo:")
# print(count_above_80_percent_max)

# print("\nCantidad de valores por debajo del 80% del valor mínimo:")
# print(count_below_80_percent_min)

# # Mostrar ejemplo con 10 valores por columna
# print("\nEjemplo con 10 valores por columna:")
# print(results['OutputFlow'].head(10))
