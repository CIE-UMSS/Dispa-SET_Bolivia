# -*- coding: utf-8 -*-
"""
Minimalist example file showing how to read the results of a Dispa-SET run

@author: Sylvain Quoilin
"""

# Add the root folder of Dispa-SET to the path so that the library can be loaded:
import sys,os
sys.path.append(os.path.abspath('../../Dispa-SET'))

# Import Dispa-SET
import dispaset as ds


# Load the inputs and the results of the simulation
inputs,results = ds.get_sim_results(path='../Simulations/bolivia_base_test_2025',cache=False)

# if needed, define the plotting range for the dispatch plot:
import pandas as pd
rng = pd.date_range(start='2025-01-01',end='2025-12-31',freq='h')

# Generate country-specific plots
ds.plot_zone(inputs,results, z='CE',rng=rng)
ds.plot_zone(inputs,results, z='NO',rng=rng)
ds.plot_zone(inputs,results, z='OR',rng=rng)
ds.plot_zone(inputs,results, z='SU',rng=rng)

# Bar plot with the installed capacities in all countries:
cap = ds.plot_zone_capacities(inputs)

# Bar plot with the energy balances in all countries:
ds.plot_energy_zone_fuel(inputs,results,ds.get_indicators_powerplant(inputs,results))

# Analyse the results for each country and provide quantitative indicators:
r = ds.get_result_analysis(inputs,results)

# Plot the reservoir levels
ds.storage_levels(inputs,results)


