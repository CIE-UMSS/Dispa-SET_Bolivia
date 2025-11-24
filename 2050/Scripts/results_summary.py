# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 19:19:31 2024

@author: navia
"""
#%% CURTAILMENT
# Obtener la suma total del curtailment
curtailment_S0 = scenarios['results_S0']['OutputCurtailedPower'].sum().sum()

# Calcular el promedio del curtailment
curtailment_avg_S0 = scenarios['results_S0']['OutputCurtailedPower'].mean().mean()

# Obtener la suma total de la generación renovable (TRG)
vregen_S0 = scenarios['df_new_plots_S0']['TRG_S0'].sum()

# Calcular el porcentaje de curtailment
curtailmentpercent_S0 = curtailment_S0 / (vregen_S0 + curtailment_S0)*100

# Mostrar los resultados
print(f"Suma total de curtailment S0: {curtailment_S0}")
print(f"Promedio de curtailment S0: {curtailment_avg_S0}")
print(f"Suma total de generación renovable S0: {vregen_S0}")
print(f"Porcentaje de curtailment S0 %: {curtailmentpercent_S0}")






# Obtener la suma total del curtailment
curtailment_S0_DI = scenarios['results_S0_DI']['OutputCurtailedPower'].sum().sum()

# Calcular el promedio del curtailment
curtailment_avg_S0_DI = scenarios['results_S0_DI']['OutputCurtailedPower'].mean().mean()

# Obtener la suma total de la generación renovable (TRG)
vregen_S0_DI = scenarios['df_new_plots_S0_DI']['TRG_S0_DI'].sum()

# Calcular el porcentaje de curtailment
curtailmentpercent_S0_DI = curtailment_S0_DI / (vregen_S0_DI + curtailment_S0_DI)*100

# Mostrar los resultados
print(f"Suma total de curtailment S0_DI: {curtailment_S0_DI}")
print(f"Promedio de curtailment S0_DI: {curtailment_avg_S0_DI}")
print(f"Suma total de generación renovable S0_DI: {vregen_S0_DI}")
print(f"Porcentaje de curtailment S0_DI %: {curtailmentpercent_S0_DI}")





# Obtener la suma total del curtailment
curtailment_S0_SI = scenarios['results_S0_SI']['OutputCurtailedPower'].sum().sum()

# Calcular el promedio del curtailment
curtailment_avg_S0_SI = scenarios['results_S0_SI']['OutputCurtailedPower'].sum().mean()

# Obtener la suma total de la generación renovable (TRG)
vregen_S0_SI = scenarios['df_new_plots_S0_SI']['TRG_S0_SI'].sum()

# Calcular el porcentaje de curtailment
curtailmentpercent_S0_SI = curtailment_S0_SI / (vregen_S0_SI + curtailment_S0_SI)*100

# Mostrar los resultados
print(f"Suma total de curtailment S0_SI: {curtailment_S0_SI}")
print(f"Promedio de curtailment S0_SI: {curtailment_avg_S0_SI}")
print(f"Suma total de generación renovable S0_SI: {vregen_S0_SI}")
print(f"Porcentaje de curtailment S0_SI %: {curtailmentpercent_S0_SI}")

#%% CO2 EMISSIONS
# Obtener la suma total del curtailment
emissions_S0 = scenarios['results_S0']['OutputEmissions'].sum().sum()

# Calcular el promedio del curtailment
emissions_avg_S0 = scenarios['results_S0']['OutputEmissions'].sum(axis=1).mean()

# Mostrar los resultados
print(f"Suma total de emissions S0: {emissions_S0}")
print(f"Promedio de emissions S0: {emissions_avg_S0}")




# Obtener la suma total del curtailment
emissions_S0_DI = scenarios['results_S0_DI']['OutputEmissions'].sum().sum()

# Calcular el promedio del curtailment
emissions_avg_S0_DI = scenarios['results_S0_DI']['OutputEmissions'].sum(axis=1).mean()

# Mostrar los resultados
print(f"Suma total de emissions S0_DI: {emissions_S0_DI}")
print(f"Promedio de emissions S0_DI: {emissions_avg_S0_DI}")




# Obtener la suma total del curtailment
emissions_S0_SI = scenarios['results_S0_SI']['OutputEmissions'].sum().sum()

# Calcular el promedio del curtailment
emissions_avg_S0_SI = scenarios['results_S0_SI']['OutputEmissions'].sum(axis=1).mean()

# Mostrar los resultados
print(f"Suma total de emissions S0_SI: {emissions_S0_SI}")
print(f"Promedio de emissions S0_SI: {emissions_avg_S0_SI}")

#%% SPILLAGE