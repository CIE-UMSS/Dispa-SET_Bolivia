# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 11:35:42 2023

@author: navia
"""

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
plt.style.use('default')
print(plt.style.available)
%matplotlib inline

#%%
# Leer el archivo CSV con encoding 'latin-1'
df = pd.read_csv('SCENARIO_5.csv', encoding='latin-1')

# Escribir el DataFrame de nuevo en un nuevo archivo CSV con encoding 'utf-8'
df.to_csv('SCENARIO_5_utf8.csv', encoding='utf-8', index=False)

#%% Load data
df0 = pd.read_csv('SCENARIO_1.csv')
df1 = pd.read_csv('SCENARIO_2.csv')
df2 = pd.read_csv('SCENARIO_3.csv')
df3 = pd.read_csv('SCENARIO_4.csv')
df4 = pd.read_csv('SCENARIO_5.csv')
df5 = pd.read_csv('SCENARIO_6.csv')

df0['TIMESTAMP']=df0['TIMESTAMP'].astype('datetime64[s]')
df0.set_index('TIMESTAMP',inplace=True, drop=True)
df0['Year']=df0.index.year
df0['Month']=df0.index.month
df0['Day']=df0.index.day
df0['Hour']=df0.index.hour
df0['Week']=df0.index.week

df1['TIMESTAMP']=df1['TIMESTAMP'].astype('datetime64[s]')
df1.set_index('TIMESTAMP',inplace=True, drop=True)
df1['Year']=df1.index.year
df1['Month']=df1.index.month
df1['Day']=df1.index.day
df1['Hour']=df1.index.hour
df1['Week']=df1.index.week

df2['TIMESTAMP']=df2['TIMESTAMP'].astype('datetime64[s]')
df2.set_index('TIMESTAMP',inplace=True, drop=True)
df2['Year']=df2.index.year
df2['Month']=df2.index.month
df2['Day']=df2.index.day
df2['Hour']=df2.index.hour
df2['Week']=df2.index.week

df3['TIMESTAMP']=df3['TIMESTAMP'].astype('datetime64[s]')
df3.set_index('TIMESTAMP',inplace=True, drop=True)
df3['Year']=df3.index.year
df3['Month']=df3.index.month
df3['Day']=df3.index.day
df3['Hour']=df3.index.hour
df3['Week']=df3.index.week

df4['TIMESTAMP']=df4['TIMESTAMP'].astype('datetime64[s]')
df4.set_index('TIMESTAMP',inplace=True, drop=True)
df4['Year']=df4.index.year
df4['Month']=df4.index.month
df4['Day']=df4.index.day
df4['Hour']=df4.index.hour
df4['Week']=df4.index.week

df5['TIMESTAMP']=df5['TIMESTAMP'].astype('datetime64[s]')
df5.set_index('TIMESTAMP',inplace=True, drop=True)
df5['Year']=df5.index.year
df5['Month']=df5.index.month
df5['Day']=df5.index.day
df5['Hour']=df5.index.hour
df5['Week']=df5.index.week
df5['Date']=df5.index.date

#%% 1. System Inertia between with and without inertia
#Plotting Penetration level histogram with seaborn
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
plt.style.use('default')
print(plt.style.available)
%matplotlib inline

plt.figure(figsize=(15,7))
sns.lineplot(data = df0, y='SystemInertia_BL1', x = 'TIMESTAMP' , linewidth=1, color='black',alpha=0.2,label='SystemInertia_BL1',n_boot=100)
sns.lineplot(data = df0, y='SystemInertia_SC1', x = 'TIMESTAMP' , linewidth=1, color='teal',alpha=1,label='SystemInertia_SC1',n_boot=100)
sns.lineplot(data = df0, y='InertiaLimit_SC1', x = 'TIMESTAMP' , linewidth=1, color='red',alpha=1,label='InertiaLimit_SC',n_boot=100)      
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize = 12)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)
plt.xlabel('TIMESTAMP', fontsize = 12)
plt.ylabel('Inertia[s]', fontsize = 12)
plt.show()

plt.figure(figsize=(15,7))
sns.lineplot(data = df1, y='SystemInertia_BL2', x = 'TIMESTAMP' , linewidth=1, color='black',alpha=0.2,label='SystemInertia_BL2',n_boot=100)
sns.lineplot(data = df1, y='SystemInertia_SC2', x = 'TIMESTAMP' , linewidth=1, color='darkslategrey',alpha=1,label='SystemInertia_SC2',n_boot=100)
sns.lineplot(data = df1, y='InertiaLimit_SC2', x = 'TIMESTAMP' , linewidth=1, color='red',alpha=1,label='InertiaLimit_SC',n_boot=100)      
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize = 12)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)
plt.xlabel('TIMESTAMP', fontsize = 12)
plt.ylabel('Inertia[s]', fontsize = 12)
plt.show()

plt.figure(figsize=(15,7))
sns.lineplot(data = df2, y='SystemInertia_BL3', x = 'TIMESTAMP' , linewidth=1, color='black',alpha=0.2,label='SystemInertia_BL3',n_boot=100)
sns.lineplot(data = df2, y='SystemInertia_SC3', x = 'TIMESTAMP' , linewidth=1, color='navy',alpha=1,label='SystemInertia_SC3',n_boot=100)
sns.lineplot(data = df2, y='InertiaLimit_SC3', x = 'TIMESTAMP' , linewidth=1, color='red',alpha=1,label='InertiaLimit_SC',n_boot=100)      
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize = 12)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)
plt.xlabel('TIMESTAMP', fontsize = 12)
plt.ylabel('Inertia[s]', fontsize = 12)
plt.show()

plt.figure(figsize=(15,7))
sns.lineplot(data = df3, y='SystemInertia_BL4', x = 'TIMESTAMP' , linewidth=1, color='black',alpha=0.2,label='SystemInertia_BL4',n_boot=100)
sns.lineplot(data = df3, y='SystemInertia_SC4', x = 'TIMESTAMP' , linewidth=1, color='seagreen',alpha=1,label='SystemInertia_SC4',n_boot=100)
sns.lineplot(data = df3, y='InertiaLimit_SC4', x = 'TIMESTAMP' , linewidth=1, color='red',alpha=1,label='InertiaLimit_SC',n_boot=100)      
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize = 12)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)
plt.xlabel('TIMESTAMP', fontsize = 12)
plt.ylabel('Inertia[s]', fontsize = 12)
plt.show()

plt.figure(figsize=(15,7))
sns.lineplot(data = df4, y='SystemInertia_BL5', x = 'TIMESTAMP' , linewidth=1, color='black',alpha=0.2,label='SystemInertia_BL5',n_boot=100)
sns.lineplot(data = df4, y='SystemInertia_SC5', x = 'TIMESTAMP' , linewidth=1, color='darkkhaki',alpha=1,label='SystemInertia_SC5',n_boot=100)
sns.lineplot(data = df4, y='InertiaLimit_SC5', x = 'TIMESTAMP' , linewidth=1, color='red',alpha=1,label='InertiaLimit_SC',n_boot=100)      
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize = 12)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)
plt.xlabel('TIMESTAMP', fontsize = 12)
plt.ylabel('Inertia[s]', fontsize = 12)
plt.show()

plt.figure(figsize=(15,7))
sns.lineplot(data = df5, y='SystemInertia_BL6', x = 'TIMESTAMP' , linewidth=1, color='black',alpha=0.2,label='SystemInertia_BL6',n_boot=100)
sns.lineplot(data = df5, y='SystemInertia_SC6', x = 'TIMESTAMP' , linewidth=1, color='plum',alpha=1,label='SystemInertia_SC6',n_boot=100)
sns.lineplot(data = df5, y='InertiaLimit_SC6', x = 'TIMESTAMP' , linewidth=1, color='red',alpha=1,label='InertiaLimit_SC',n_boot=100)      
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize = 12)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)
plt.xlabel('TIMESTAMP', fontsize = 12)
plt.ylabel('Inertia[s]', fontsize = 12)
plt.show()
#%%
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
plt.style.use('default')
%matplotlib inline

# Supongamos que tienes una lista de DataFrames llamada df_list
df_list = [df0, df1, df2, df3, df4, df5]

# Tamaño de la cuadrícula
grid_size = (2, 3)  # 2 filas, 3 columnas

# Crear subgráficos
fig, axes = plt.subplots(nrows=grid_size[0], ncols=grid_size[1], figsize=(18, 10))

# Aplanar la matriz de subgráficos para facilitar el índice
axes = axes.flatten()

# Colores personalizados para cada línea
line_colors = ['black', 'darkslategrey', 'navy', 'seagreen', 'darkkhaki', 'plum']

# Iterar a través de DataFrames y subgráficos
for i, (df, ax) in enumerate(zip(df_list, axes)):
    sns.lineplot(data=df, y=f'SystemInertia_BL{i + 1}', x='TIMESTAMP', linewidth=1, color=line_colors[i], alpha=0.2, label=f'SystemInertia_BL{i + 1}', n_boot=100, ax=ax)
    sns.lineplot(data=df, y=f'SystemInertia_SC{i + 1}', x='TIMESTAMP', linewidth=1, color=line_colors[i], alpha=1, label=f'SystemInertia_SC{i + 1}', n_boot=100, ax=ax)
    sns.lineplot(data=df, y=f'InertiaLimit_SC{i + 1}', x='TIMESTAMP', linewidth=1, color='red', alpha=1, label='InertiaLimit_SC', n_boot=100, ax=ax)
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=12)
    ax.set_xticklabels(ax.get_xticks(), fontsize=10)
    ax.set_yticklabels(ax.get_yticks(), fontsize=10)
    ax.set_xlabel('TIMESTAMP', fontsize=12)
    ax.set_ylabel('Inertia[s]', fontsize=12)

# Ajustar diseño
plt.tight_layout()
plt.show()

#%% 1. System Inertia between with and without inertia
#Plotting Penetration level histogram with seaborn
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
plt.style.use('default')
print(plt.style.available)
%matplotlib inline

plt.figure(figsize=(15,7))
sns.lineplot(data = df0, y='SystemInertia_BL1', x = 'TIMESTAMP' , linewidth=1, color='black',alpha=0.2,label='SystemInertia_BL1',n_boot=100)
sns.lineplot(data = df0, y='SystemInertia_MC1', x = 'TIMESTAMP' , linewidth=1, color='teal',alpha=1,label='SystemInertia_MC1',n_boot=100)
sns.lineplot(data = df0, y='InertiaLimit_MC1', x = 'TIMESTAMP' , linewidth=1, color='red',alpha=1,label='InertiaLimit_MC',n_boot=100)      
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize = 12)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)
plt.xlabel('TIMESTAMP', fontsize = 12)
plt.ylabel('Inertia[s]', fontsize = 12)
plt.show()

plt.figure(figsize=(15,7))
sns.lineplot(data = df1, y='SystemInertia_BL2', x = 'TIMESTAMP' , linewidth=1, color='black',alpha=0.2,label='SystemInertia_BL2',n_boot=100)
sns.lineplot(data = df1, y='SystemInertia_MC2', x = 'TIMESTAMP' , linewidth=1, color='darkslategrey',alpha=1,label='SystemInertia_MC2',n_boot=100)
sns.lineplot(data = df1, y='InertiaLimit_MC2', x = 'TIMESTAMP' , linewidth=1, color='red',alpha=1,label='InertiaLimit_MC',n_boot=100)      
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize = 12)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)
plt.xlabel('TIMESTAMP', fontsize = 12)
plt.ylabel('Inertia[s]', fontsize = 12)
plt.show()

plt.figure(figsize=(15,7))
sns.lineplot(data = df2, y='SystemInertia_BL3', x = 'TIMESTAMP' , linewidth=1, color='black',alpha=0.2,label='SystemInertia_BL3',n_boot=100)
sns.lineplot(data = df2, y='SystemInertia_MC3', x = 'TIMESTAMP' , linewidth=1, color='navy',alpha=1,label='SystemInertia_MC3',n_boot=100)
sns.lineplot(data = df2, y='InertiaLimit_MC3', x = 'TIMESTAMP' , linewidth=1, color='red',alpha=1,label='InertiaLimit_MC',n_boot=100)      
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize = 12)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)
plt.xlabel('TIMESTAMP', fontsize = 12)
plt.ylabel('Inertia[s]', fontsize = 12)
plt.show()

plt.figure(figsize=(15,7))
sns.lineplot(data = df3, y='SystemInertia_BL4', x = 'TIMESTAMP' , linewidth=1, color='black',alpha=0.2,label='SystemInertia_BL4',n_boot=100)
sns.lineplot(data = df3, y='SystemInertia_MC4', x = 'TIMESTAMP' , linewidth=1, color='seagreen',alpha=1,label='SystemInertia_MC4',n_boot=100)
sns.lineplot(data = df3, y='InertiaLimit_MC4', x = 'TIMESTAMP' , linewidth=1, color='red',alpha=1,label='InertiaLimit_MC',n_boot=100)      
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize = 12)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)
plt.xlabel('TIMESTAMP', fontsize = 12)
plt.ylabel('Inertia[s]', fontsize = 12)
plt.show()

plt.figure(figsize=(15,7))
sns.lineplot(data = df4, y='SystemInertia_BL5', x = 'TIMESTAMP' , linewidth=1, color='black',alpha=0.2,label='SystemInertia_BL5',n_boot=100)
sns.lineplot(data = df4, y='SystemInertia_MC5', x = 'TIMESTAMP' , linewidth=1, color='darkkhaki',alpha=1,label='SystemInertia_MC5',n_boot=100)
sns.lineplot(data = df4, y='InertiaLimit_MC5', x = 'TIMESTAMP' , linewidth=1, color='red',alpha=1,label='InertiaLimit_MC',n_boot=100)      
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize = 12)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)
plt.xlabel('TIMESTAMP', fontsize = 12)
plt.ylabel('Inertia[s]', fontsize = 12)
plt.show()

plt.figure(figsize=(15,7))
sns.lineplot(data = df5, y='SystemInertia_BL6', x = 'TIMESTAMP' , linewidth=1, color='black',alpha=0.2,label='SystemInertia_BL6',n_boot=100)
sns.lineplot(data = df5, y='SystemInertia_MC6', x = 'TIMESTAMP' , linewidth=1, color='plum',alpha=1,label='SystemInertia_MC6',n_boot=100)
sns.lineplot(data = df5, y='InertiaLimit_MC6', x = 'TIMESTAMP' , linewidth=1, color='red',alpha=1,label='InertiaLimit_MC',n_boot=100)      
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize = 12)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)
plt.xlabel('TIMESTAMP', fontsize = 12)
plt.ylabel('Inertia[s]', fontsize = 12)
plt.show()
#%%
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
plt.style.use('default')
%matplotlib inline

# Supongamos que tienes una lista de DataFrames llamada df_list
df_list = [df0, df1, df2, df3, df4, df5]

# Tamaño de la cuadrícula
grid_size = (2, 3)  # 2 filas, 3 columnas

# Crear subgráficos
fig, axes = plt.subplots(nrows=grid_size[0], ncols=grid_size[1], figsize=(18, 10))

# Aplanar la matriz de subgráficos para facilitar el índice
axes = axes.flatten()

# Colores personalizados para cada línea
line_colors = ['black', 'darkslategrey', 'navy', 'seagreen', 'darkkhaki', 'plum']

# Iterar a través de DataFrames y subgráficos
for i, (df, ax) in enumerate(zip(df_list, axes)):
    sns.lineplot(data=df, y=f'SystemInertia_BL{i + 1}', x='TIMESTAMP', linewidth=1, color=line_colors[i], alpha=0.2, label=f'SystemInertia_BL{i + 1}', n_boot=100, ax=ax)
    sns.lineplot(data=df, y=f'SystemInertia_MC{i + 1}', x='TIMESTAMP', linewidth=1, color=line_colors[i], alpha=1, label=f'SystemInertia_MC{i + 1}', n_boot=100, ax=ax)
    sns.lineplot(data=df, y=f'InertiaLimit_MC{i + 1}', x='TIMESTAMP', linewidth=1, color='red', alpha=1, label='InertiaLimit_MC', n_boot=100, ax=ax)
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=12)
    ax.set_xticklabels(ax.get_xticks(), fontsize=10)
    ax.set_yticklabels(ax.get_yticks(), fontsize=10)
    ax.set_xlabel('TIMESTAMP', fontsize=12)
    ax.set_ylabel('Inertia[s]', fontsize=12)

# Ajustar diseño
plt.tight_layout()
plt.show()

#%% 2. Penetration level between scenarios (ALL THE SCENARIOS IN ONE PLOT)
#Plotting Penetration level histogram with seaborn
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
plt.style.use('default')
print(plt.style.available)
%matplotlib inline
fig, ax = plt.subplots(figsize=(15,7))
sns.lineplot(data = df0, y='PL_BL1', x = 'Week' , linewidth=1, color='teal',alpha=1,label='Penetration_Level_BL1')
sns.lineplot(data = df1, y='PL_BL2', x = 'Week' , linewidth=1, color='darkslategrey',alpha=1,label='Penetration_Level_BL2')
sns.lineplot(data = df2, y='PL_BL3', x = 'Week' , linewidth=1, color='navy',alpha=1,label='Penetration_Level_BL3')
sns.lineplot(data = df3, y='PL_BL4', x = 'Week' , linewidth=1, color='seagreen',alpha=1,label='Penetration_Level_BL4')
sns.lineplot(data = df4, y='PL_BL5', x = 'Week' , linewidth=1, color='darkkhaki',alpha=1,label='Penetration_Level_BL4')
sns.lineplot(data = df5, y='PL_BL6', x = 'Week' , linewidth=1, color='plum',alpha=1,label='Penetration_Level_BL6')      
sns.lineplot(data = df0, y='PL_SC1', x = 'Week' , linewidth=1, color='teal',alpha=1,label='Penetration_Level_SC1')
sns.lineplot(data = df1, y='PL_SC2', x = 'Week' , linewidth=1, color='darkslategrey',alpha=1,label='Penetration_Level_SC2')
sns.lineplot(data = df2, y='PL_SC3', x = 'Week' , linewidth=1, color='navy',alpha=1,label='Penetration_Level_SC3')
sns.lineplot(data = df3, y='PL_SC4', x = 'Week' , linewidth=1, color='seagreen',alpha=1,label='Penetration_Level_SC4')
sns.lineplot(data = df4, y='PL_SC5', x = 'Week' , linewidth=1, color='darkkhaki',alpha=1,label='Penetration_Level_SC5')
sns.lineplot(data = df5, y='PL_SC6', x = 'Week' , linewidth=1, color='plum',alpha=1,label='Penetration_Level_SC6')      
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize = 22)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.xlabel('Week', fontsize = 22)
plt.ylabel('Penetration Level[pu]', fontsize = 22)
plt.show()
#%% 2. Penetration level between scenarios (ALL THE SCENARIOS IN ONE PLOT)
#Plotting Penetration level histogram with seaborn
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
plt.style.use('default')
print(plt.style.available)
%matplotlib inline
fig, ax = plt.subplots(figsize=(15,7))
sns.lineplot(data = df0, y='PL_BL1', x = 'Week' , linewidth=1, color='teal',alpha=1,label='Penetration_Level_BL1')
sns.lineplot(data = df1, y='PL_BL2', x = 'Week' , linewidth=1, color='darkslategrey',alpha=1,label='Penetration_Level_BL2')
sns.lineplot(data = df2, y='PL_BL3', x = 'Week' , linewidth=1, color='navy',alpha=1,label='Penetration_Level_BL3')
sns.lineplot(data = df3, y='PL_BL4', x = 'Week' , linewidth=1, color='seagreen',alpha=1,label='Penetration_Level_BL4')
sns.lineplot(data = df4, y='PL_BL5', x = 'Week' , linewidth=1, color='darkkhaki',alpha=1,label='Penetration_Level_BL5')
sns.lineplot(data = df5, y='PL_BL6', x = 'Week' , linewidth=1, color='plum',alpha=1,label='Penetration_Level_BL6')      
sns.lineplot(data = df0, y='PL_MC1', x = 'Week' , linewidth=1, color='teal',alpha=1,label='Penetration_Level_MC1')
sns.lineplot(data = df1, y='PL_MC2', x = 'Week' , linewidth=1, color='darkslategrey',alpha=1,label='Penetration_Level_MC2')
sns.lineplot(data = df2, y='PL_MC3', x = 'Week' , linewidth=1, color='navy',alpha=1,label='Penetration_Level_MC3')
sns.lineplot(data = df3, y='PL_MC4', x = 'Week' , linewidth=1, color='seagreen',alpha=1,label='Penetration_Level_MC4')
sns.lineplot(data = df4, y='PL_MC5', x = 'Week' , linewidth=1, color='darkkhaki',alpha=1,label='Penetration_Level_MC5')
sns.lineplot(data = df5, y='PL_MC6', x = 'Week' , linewidth=1, color='plum',alpha=1,label='Penetration_Level_MC6')      
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize = 22)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.xlabel('Week', fontsize = 22)
plt.ylabel('Penetration Level[pu]', fontsize = 22)
plt.show()


#%% 3. Penetration level inertia constraint (DIFFERENT PLOTS FOR EACH SCENARIO)
#Plotting Penetration level histogram with seaborn
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
plt.style.use('default')
print(plt.style.available)
%matplotlib inline

fig, ax = plt.subplots(figsize=(15,7))
sns.lineplot(data = df0, y='PL_SG0', x = 'Week' , linewidth=1, color='black',alpha=1,label='Penetration_Level_S0')
sns.lineplot(data = df0, y='PL_MG0', x = 'Week' , linewidth=1, color='teal',alpha=1,label='Penetration_Level_M0')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize = 12)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)
plt.xlabel('Week', fontsize = 12)
plt.ylabel('Penetration Level[pu]', fontsize = 12)
plt.show()

fig, ax = plt.subplots(figsize=(15,7))
sns.lineplot(data = df1, y='PL_SG1', x = 'Week' , linewidth=1, color='black',alpha=1,label='Penetration_Level_S1')
sns.lineplot(data = df1, y='PL_MG1', x = 'Week' , linewidth=1, color='darkslategrey',alpha=1,label='Penetration_Level_M1')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize = 12)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)
plt.xlabel('Week', fontsize = 12)
plt.ylabel('Penetration Level[pu]', fontsize = 12)
plt.show()

fig, ax = plt.subplots(figsize=(15,7))
sns.lineplot(data = df2, y='PL_SG2', x = 'Week' , linewidth=1, color='black',alpha=1,label='Penetration_Level_S2')
sns.lineplot(data = df2, y='PL_MG2', x = 'Week' , linewidth=1, color='navy',alpha=1,label='Penetration_Level_M2')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize = 12)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)
plt.xlabel('Week', fontsize = 12)
plt.ylabel('Penetration Level[pu]', fontsize = 12)
plt.show()

fig, ax = plt.subplots(figsize=(15,7))
sns.lineplot(data = df3, y='PL_SG3', x = 'Week' , linewidth=1, color='black',alpha=1,label='Penetration_Level_S3')
sns.lineplot(data = df3, y='PL_MG3', x = 'Week' , linewidth=1, color='seagreen',alpha=1,label='Penetration_Level_M3')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize = 12)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)
plt.xlabel('Week', fontsize = 12)
plt.ylabel('Penetration Level[pu]', fontsize = 12)
plt.show()

fig, ax = plt.subplots(figsize=(15,7))
sns.lineplot(data = df4, y='PL_SG4', x = 'Week' , linewidth=1, color='black',alpha=1,label='Penetration_Level_S4')
sns.lineplot(data = df4, y='PL_MG4', x = 'Week' , linewidth=1, color='darkkhaki',alpha=1,label='Penetration_Level_M4')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize = 12)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)
plt.xlabel('Week', fontsize = 12)
plt.ylabel('Penetration Level[pu]', fontsize = 12)
plt.show()

fig, ax = plt.subplots(figsize=(15,7))
sns.lineplot(data = df5, y='PL_SG5', x = 'Week' , linewidth=1, color='black',alpha=1,label='Penetration_Level_S5')      
sns.lineplot(data = df5, y='PL_MG5', x = 'Week' , linewidth=1, color='plum',alpha=1,label='Penetration_Level_M5')      
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize = 12)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)
plt.xlabel('Week', fontsize = 12)
plt.ylabel('Penetration Level[pu]', fontsize = 12)
plt.show()

#%% 4. Curtailment between scenarios (ALL THE SCENARIOS IN ONE PLOT)
#Plotting Penetration level histogram with seaborn
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
plt.style.use('default')
print(plt.style.available)
%matplotlib inline
fig, ax = plt.subplots(figsize=(15,7))
sns.lineplot(data = df0, y='Curtailment_BL1', x = 'Week' , linewidth=1, color='teal',alpha=1,label='Curtailment_BL1')
sns.lineplot(data = df1, y='Curtailment_BL2', x = 'Week' , linewidth=1, color='darkslategrey',alpha=1,label='Curtailment_BL2')
sns.lineplot(data = df2, y='Curtailment_BL3', x = 'Week' , linewidth=1, color='navy',alpha=1,label='Curtailment_BL3')
sns.lineplot(data = df3, y='Curtailment_BL4', x = 'Week' , linewidth=1, color='seagreen',alpha=1,label='Curtailment_BL4')
sns.lineplot(data = df4, y='Curtailment_BL5', x = 'Week' , linewidth=1, color='darkkhaki',alpha=1,label='Curtailment_BL5')
sns.lineplot(data = df5, y='Curtailment_BL6', x = 'Week' , linewidth=1, color='plum',alpha=1,label='Curtailment_BL6')      
sns.lineplot(data = df0, y='Curtailment_SC1', x = 'Week' , linewidth=1, color='teal',alpha=1,label='Curtailment_SC1')
sns.lineplot(data = df1, y='Curtailment_SC2', x = 'Week' , linewidth=1, color='darkslategrey',alpha=1,label='Curtailment_SC2')
sns.lineplot(data = df2, y='Curtailment_SC3', x = 'Week' , linewidth=1, color='navy',alpha=1,label='Curtailment_SC3')
sns.lineplot(data = df3, y='Curtailment_SC4', x = 'Week' , linewidth=1, color='seagreen',alpha=1,label='Curtailment_SC4')
sns.lineplot(data = df4, y='Curtailment_SC5', x = 'Week' , linewidth=1, color='darkkhaki',alpha=1,label='Curtailment_SC5')
sns.lineplot(data = df5, y='Curtailment_SC6', x = 'Week' , linewidth=1, color='plum',alpha=1,label='Curtailment_SC6')      
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize = 22)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.xlabel('Week', fontsize = 22)
plt.ylabel('Curtailment[MWh]', fontsize = 22)
plt.show()
#%% 4. Curtailment between scenarios (ALL THE SCENARIOS IN ONE PLOT)
#Plotting Penetration level histogram with seaborn
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
plt.style.use('default')
print(plt.style.available)
%matplotlib inline
fig, ax = plt.subplots(figsize=(15,7))
sns.lineplot(data = df0, y='Curtailment_BL1', x = 'Week' , linewidth=1, color='teal',alpha=1,label='Curtailment_BL1')
sns.lineplot(data = df1, y='Curtailment_BL2', x = 'Week' , linewidth=1, color='darkslategrey',alpha=1,label='Curtailment_BL2')
sns.lineplot(data = df2, y='Curtailment_BL3', x = 'Week' , linewidth=1, color='navy',alpha=1,label='Curtailment_BL3')
sns.lineplot(data = df3, y='Curtailment_BL4', x = 'Week' , linewidth=1, color='seagreen',alpha=1,label='Curtailment_BL4')
sns.lineplot(data = df4, y='Curtailment_BL5', x = 'Week' , linewidth=1, color='darkkhaki',alpha=1,label='Curtailment_BL5')
sns.lineplot(data = df5, y='Curtailment_BL6', x = 'Week' , linewidth=1, color='plum',alpha=1,label='Curtailment_BL6')      
sns.lineplot(data = df0, y='Curtailment_MC1', x = 'Week' , linewidth=1, color='teal',alpha=1,label='Curtailment_MC1')
sns.lineplot(data = df1, y='Curtailment_MC2', x = 'Week' , linewidth=1, color='darkslategrey',alpha=1,label='Curtailment_MC2')
sns.lineplot(data = df2, y='Curtailment_MC3', x = 'Week' , linewidth=1, color='navy',alpha=1,label='Curtailment_MC3')
sns.lineplot(data = df3, y='Curtailment_MC4', x = 'Week' , linewidth=1, color='seagreen',alpha=1,label='Curtailment_MC4')
sns.lineplot(data = df4, y='Curtailment_MC5', x = 'Week' , linewidth=1, color='darkkhaki',alpha=1,label='Curtailment_MC5')
sns.lineplot(data = df5, y='Curtailment_MC6', x = 'Week' , linewidth=1, color='plum',alpha=1,label='Curtailment_MC6')      
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize = 22)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.xlabel('Week', fontsize = 22)
plt.ylabel('Curtailment[MWh]', fontsize = 22)
plt.show()
#%% 4. Curtailment  (DIFFERENT PLOTS FOR EACH SCENARIO)
#Plotting Penetration level histogram with seaborn
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
plt.style.use('default')
print(plt.style.available)
%matplotlib inline

fig, ax = plt.subplots(figsize=(15,7))
sns.lineplot(data = df0, y='Curtailment_S0', x = 'Week' , linewidth=1, color='black',alpha=1,label='Curtailment_S0')
sns.lineplot(data = df0, y='Curtailment_M0', x = 'Week' , linewidth=1, color='teal',alpha=1,label='Curtailment_M0')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize = 12)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)
plt.xlabel('Week', fontsize = 12)
plt.ylabel('Curtailment[MWh]', fontsize = 12)
plt.show()

fig, ax = plt.subplots(figsize=(15,7))
sns.lineplot(data = df1, y='Curtailment_S1', x = 'Week' , linewidth=1, color='black',alpha=1,label='Curtailment_S1')
sns.lineplot(data = df1, y='Curtailment_M1', x = 'Week' , linewidth=1, color='darkslategrey',alpha=1,label='Curtailment_M1')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize = 12)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)
plt.xlabel('Week', fontsize = 12)
plt.ylabel('Curtailment[MWh]', fontsize = 12)
plt.show()

fig, ax = plt.subplots(figsize=(15,7))
sns.lineplot(data = df2, y='Curtailment_S2', x = 'Week' , linewidth=1, color='black',alpha=1,label='Curtailment_S2')
sns.lineplot(data = df2, y='Curtailment_M2', x = 'Week' , linewidth=1, color='navy',alpha=1,label='Curtailment_M2')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize = 12)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)
plt.xlabel('Week', fontsize = 12)
plt.ylabel('Curtailment[MWh]', fontsize = 12)
plt.show()

fig, ax = plt.subplots(figsize=(15,7))
sns.lineplot(data = df3, y='Curtailment_S3', x = 'Week' , linewidth=1, color='black',alpha=1,label='Curtailment_S3')
sns.lineplot(data = df3, y='Curtailment_M3', x = 'Week' , linewidth=1, color='seagreen',alpha=1,label='Curtailment_M3')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize = 12)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)
plt.xlabel('Week', fontsize = 12)
plt.ylabel('Curtailment[MWh]', fontsize = 12)
plt.show()

fig, ax = plt.subplots(figsize=(15,7))
sns.lineplot(data = df4, y='Curtailment_S4', x = 'Week' , linewidth=1, color='black',alpha=1,label='Curtailment_S4')
sns.lineplot(data = df4, y='Curtailment_M4', x = 'Week' , linewidth=1, color='darkkhaki',alpha=1,label='Curtailment_M4')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize = 12)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)
plt.xlabel('Week', fontsize = 12)
plt.ylabel('Curtailment[MWh]', fontsize = 12)
plt.show()

fig, ax = plt.subplots(figsize=(15,7))
sns.lineplot(data = df5, y='Curtailment_S5', x = 'Week' , linewidth=1, color='black',alpha=1,label='Curtailment_S5')      
sns.lineplot(data = df5, y='Curtailment_M5', x = 'Week' , linewidth=1, color='plum',alpha=1,label='Curtailment_M5')      
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize = 12)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)
plt.xlabel('Week', fontsize = 12)
plt.ylabel('Curtailment[MWh]', fontsize = 12)
plt.show()

#%% 5. System Inertia between with and without inertia
#Plotting Penetration level histogram with seaborn
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
plt.style.use('default')
print(plt.style.available)
%matplotlib inline

plt.figure(figsize=(15,7))
sns.lineplot(data = df0, y='Inertia_S0', x = 'TIMESTAMP' , linewidth=1, color='teal',alpha=0.2,label='Inertia_S0',n_boot=100)
sns.lineplot(data = df0, y='Inertia_M0', x = 'TIMESTAMP' , linewidth=1, color='teal',alpha=1,label='Inertia_M0',n_boot=100)
sns.lineplot(data = df1, y='Inertia_S1', x = 'TIMESTAMP' , linewidth=1, color='darkslategrey',alpha=0.2,label='Inertia_S1',n_boot=100)
sns.lineplot(data = df1, y='Inertia_M1', x = 'TIMESTAMP' , linewidth=1, color='darkslategrey',alpha=1,label='Inertia_M1',n_boot=100)
sns.lineplot(data = df2, y='Inertia_S2', x = 'TIMESTAMP' , linewidth=1, color='navy',alpha=0.2,label='Inertia_S2',n_boot=100)
sns.lineplot(data = df2, y='Inertia_M2', x = 'TIMESTAMP' , linewidth=1, color='navy',alpha=1,label='Inertia_M2',n_boot=100)
sns.lineplot(data = df3, y='Inertia_S3', x = 'TIMESTAMP' , linewidth=1, color='seagreen',alpha=0.2,label='Inertia_S3',n_boot=100)
sns.lineplot(data = df3, y='Inertia_M3', x = 'TIMESTAMP' , linewidth=1, color='seagreen',alpha=1,label='Inertia_M3',n_boot=100)
sns.lineplot(data = df4, y='Inertia_S4', x = 'TIMESTAMP' , linewidth=1, color='darkkhaki',alpha=0.2,label='Inertia_S4',n_boot=100)
sns.lineplot(data = df4, y='Inertia_M4', x = 'TIMESTAMP' , linewidth=1, color='darkkhaki',alpha=1,label='Inertia_S4',n_boot=100)
sns.lineplot(data = df5, y='Inertia_S5', x = 'TIMESTAMP' , linewidth=1, color='plum',alpha=0.2,label='Inertia_S5',n_boot=100)
sns.lineplot(data = df5, y='Inertia_M5', x = 'TIMESTAMP' , linewidth=1, color='plum',alpha=1,label='Inertia_M5',n_boot=100)
sns.lineplot(data = df0, y='Hlimit', x = 'TIMESTAMP' , linewidth=1, color='red',alpha=1,label='Inertia_Limit',n_boot=100)      
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize = 22)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.xlabel('TIMESTAMP', fontsize = 22)
plt.ylabel('Inertia[s]', fontsize = 22)
plt.show()

#%% 5. System Inertia without inertia constraint
#Plotting Penetration level histogram with seaborn
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
plt.style.use('default')
print(plt.style.available)
%matplotlib inline

plt.figure(figsize=(15,7))
sns.lineplot(data = df0, y='SystemInertia_BL1', x = 'TIMESTAMP' , linewidth=1, color='teal',alpha=1,label='SystemInertia_BL1',n_boot=100)
sns.lineplot(data = df1, y='SystemInertia_BL2', x = 'TIMESTAMP' , linewidth=1, color='darkslategrey',alpha=1,label='ISystemInertia_BL2',n_boot=100)
sns.lineplot(data = df2, y='SystemInertia_BL3', x = 'TIMESTAMP' , linewidth=1, color='navy',alpha=1,label='SystemInertia_BL3',n_boot=100)
sns.lineplot(data = df3, y='SystemInertia_BL4', x = 'TIMESTAMP' , linewidth=1, color='seagreen',alpha=1,label='SystemInertia_BL4',n_boot=100)
sns.lineplot(data = df4, y='SystemInertia_BL5', x = 'TIMESTAMP' , linewidth=1, color='darkkhaki',alpha=1,label='SystemInertia_BL5',n_boot=100)
sns.lineplot(data = df5, y='SystemInertia_BL6', x = 'TIMESTAMP' , linewidth=1, color='plum',alpha=1,label='SystemInertia_BL6',n_boot=100)
sns.lineplot(data = df0, y='InertiaLimit_SC1', x = 'TIMESTAMP' , linewidth=1, color='red',alpha=1,label='InertiaLimit_SC1',n_boot=100)      
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize = 22)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.xlabel('TIMESTAMP', fontsize = 22)
plt.ylabel('Inertia[s]', fontsize = 22)
plt.show()

#%% 6. System Inertia with inertia constraint
#Plotting Penetration level histogram with seaborn
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
plt.style.use('default')
print(plt.style.available)
%matplotlib inline

plt.figure(figsize=(15,7))
sns.lineplot(data = df0, y='SystemInertia_SC1', x = 'TIMESTAMP' , linewidth=1, color='teal',alpha=1,label='SystemInertia_SC1',n_boot=100)
sns.lineplot(data = df1, y='SystemInertia_SC2', x = 'TIMESTAMP' , linewidth=1, color='darkslategrey',alpha=1,label='SystemInertia_SC2',n_boot=100)
sns.lineplot(data = df2, y='SystemInertia_SC3', x = 'TIMESTAMP' , linewidth=1, color='navy',alpha=1,label='SystemInertia_SC3',n_boot=100)
sns.lineplot(data = df3, y='SystemInertia_SC4', x = 'TIMESTAMP' , linewidth=1, color='seagreen',alpha=1,label='SystemInertia_SC4',n_boot=100)
sns.lineplot(data = df4, y='SystemInertia_SC5', x = 'TIMESTAMP' , linewidth=1, color='darkkhaki',alpha=1,label='SystemInertia_SC5',n_boot=100)
sns.lineplot(data = df5, y='SystemInertia_SC6', x = 'TIMESTAMP' , linewidth=1, color='plum',alpha=1,label='SystemInertia_SC6',n_boot=100)
sns.lineplot(data = df0, y='InertiaLimit_SC1', x = 'TIMESTAMP' , linewidth=1, color='red',alpha=1,label='InertiaLimit_SC1',n_boot=100)      
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize = 22)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.xlabel('TIMESTAMP', fontsize = 22)
plt.ylabel('Inertia[s]', fontsize = 22)
plt.show()


#%% 7. Distribucion acumulada
############################################   
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

plt.style.use('default')
print(plt.style.available)
%matplotlib inline

#No. of data points used
N = 8761

df = pd.read_csv('INERTIA.1.csv',index_col=(0))
plt.figure(figsize=(15, 7), dpi=150, edgecolor='yellow')

#sort data in ascending order
Inertia_S0 = np.sort(df['SystemInertia_BL1'])
Inertia_S1 = np.sort(df['SystemInertia_BL2'])
Inertia_S2 = np.sort(df['SystemInertia_BL3'])
Inertia_S3 = np.sort(df['SystemInertia_BL4'])
Inertia_S4 = np.sort(df['SystemInertia_BL5'])
Inertia_S5 = np.sort(df['SystemInertia_BL6'])

Inertia_M0 = np.sort(df['SystemInertia_SC1'])
Inertia_M1 = np.sort(df['SystemInertia_SC2'])
Inertia_M2 = np.sort(df['SystemInertia_SC3'])
Inertia_M3 = np.sort(df['SystemInertia_SC4'])
Inertia_M4 = np.sort(df['SystemInertia_SC5'])
Inertia_M5 = np.sort(df['SystemInertia_SC6'])

Mlimit = np.sort(df['InertiaLimit_SC'])

# get the cdf values of y
y = np.arange(N) / float(N)

# adding title to the plot
# plt.title('Cumulative Distribution of System Inertia', fontsize = 30)  

# plotting
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.xlabel('Inertia(H)[s]', fontsize = 22)
plt.ylabel('Distribution', fontsize = 22)
  
  
plt.plot(Inertia_S0, y, marker='o', alpha=0.1, color='teal')
plt.plot(Inertia_S1, y, marker='o', alpha=0.1, color='darkslategrey')
plt.plot(Inertia_S2, y, marker='o', alpha=0.1, color='navy')
plt.plot(Inertia_S3, y, marker='o', alpha=0.1, color='seagreen')
plt.plot(Inertia_S4, y, marker='o', alpha=0.1, color='darkkhaki')
plt.plot(Inertia_S5, y, marker='o', alpha=0.1, color='plum')

plt.plot(Inertia_M0, y, marker='x', alpha=0.1, color='teal')
plt.plot(Inertia_M1, y, marker='x', alpha=0.1, color='darkslategrey')
plt.plot(Inertia_M2, y, marker='x', alpha=0.1, color='navy')
plt.plot(Inertia_M3, y, marker='x', alpha=0.1, color='seagreen')
plt.plot(Inertia_M4, y, marker='x', alpha=0.1, color='darkkhaki')
plt.plot(Inertia_M5, y, marker='x', alpha=0.1, color='plum')

plt.plot(Mlimit, y, marker='o', alpha=1, color='red')
# adding legend to the curve
plt.legend([
            'Inertia_S0',
            'Inertia_S1',
            'Inertia_S2',
            'Inertia_S3',
            'Inertia_S4',
            'Inertia_S5',
            'Inertia_M0',
            'Inertia_M1',
            'Inertia_M2',
            'Inertia_M3',
            'Inertia_M4',
            'Inertia_M5', 
            'Mlimit'
            ], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize = 22)

#%% 8. Distribucion acumulada en boxplots
############################################  
import matplotlib.pyplot as plt
import pandas as pd
plt.style.use('default')
print(plt.style.available)
%matplotlib inline



df = pd.read_csv('INERTIA.1.csv',index_col=(0))
plt.figure(figsize=(15, 7), dpi=150, edgecolor='yellow')




# adding title to the plot
# plt.title('Cumulative Distribution of System Inertia', fontsize = 30)  

# plotting
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.xlabel('Inertia(H)[s]', fontsize = 22)
plt.ylabel('Distribution', fontsize = 22)
  
# adding legend to the curve
plt.legend([
            'Inertia_S0',
            'Inertia_S1',
            'Inertia_S2',
            'Inertia_S3',
            'Inertia_S4',
            'Inertia_S5',
            'Inertia_M0',
            'Inertia_M1',
            'Inertia_M2',
            'Inertia_M3',
            'Inertia_M4',
            'Inertia_M5', 
            'Mlimit'
            ], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize = 22)
for i in df.columns:
    df.boxplot(column=i)
plt.show()
#%% 8. Distribucion acumulada en boxplots
############################################  
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv('INERTIA.1.csv',index_col=(0))

sns.set(style='whitegrid')

fig, ax = plt.subplots(figsize=(25, 9))
xvalues = [
'SystemInertia_BL1',
'SystemInertia_SC1',
'SystemInertia_MC1',
'SystemInertia_BL2',
'SystemInertia_SC2',
'SystemInertia_MC2',
'SystemInertia_BL3',
'SystemInertia_SC3',
'SystemInertia_MC3',
'SystemInertia_BL4',
'SystemInertia_SC4',
'SystemInertia_MC4',
'SystemInertia_BL5',
'SystemInertia_SC5',
'SystemInertia_MC5',
'SystemInertia_BL6',
'SystemInertia_SC6',
'SystemInertia_MC6'       
]

palette = ['teal','teal','teal','darkslategrey','darkslategrey','darkslategrey','navy','navy','navy', 'seagreen', 'seagreen', 'seagreen', 'darkkhaki', 'darkkhaki', 'darkkhaki','plum','plum','plum']

melted_df = df.set_axis(xvalues, axis=1).melt(var_name='Variable', value_name='Accuracy')
sns.boxplot(data=melted_df, x='Variable', y='Accuracy', hue='Variable', palette=palette,
            width=0.7, dodge=False, ax=ax)

ax.legend_.remove()  # remove the legend, as the information is already present in the x labels
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)

# Modify x-axis labels to keep only the last 3 characters
new_labels = [label[-3:] for label in xvalues]
ax.set_xticklabels(new_labels, fontsize=20)

ax.set_xlabel('', fontsize=20)  # remove unuseful xlabel ('Variable')
ax.set_ylabel("Inertia[s]", fontsize=20)
sns.despine(top=True, right=True, left=True, bottom=False)

plt.legend([
            'SystemInertia_BL1',
            'SystemInertia_BL2',
            'SystemInertia_BL3',
            'SystemInertia_BL4',
            'SystemInertia_BL5',
            'SystemInertia_BL6',
            'SystemInertia_SC1',
            'SystemInertia_SC2',
            'SystemInertia_SC3',
            'SystemInertia_SC4',
            'SystemInertia_SC5',
            'SystemInertia_SC6',
            'SystemInertia_MC1',
            'SystemInertia_MC2',
            'SystemInertia_MC3',
            'SystemInertia_MC4',
            'SystemInertia_MC5',
            'SystemInertia_MC6' 
            ], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize = 22)

#%% 8. Distribucion acumulada en boxplots
############################################  
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv('INERTIA.1.csv',index_col=(0))
sns.set(style='whitegrid')

fig, ax = plt.subplots(figsize=(25, 9))
xvalues = [
'Mlimit',
'Inertia_S0',
'Inertia_M0',
'Inertia_S1',
'Inertia_M1',
'Inertia_S2',
'Inertia_M2',
'Inertia_S3',
'Inertia_M3',
'Inertia_S4',
'Inertia_M4',
'Inertia_S5',
'Inertia_M5'       
]

palette = ['red','grey','teal','grey','darkslategrey','grey','navy', 'grey', 'seagreen', 'grey', 'darkkhaki','grey','plum']

melted_df = df.set_axis(xvalues, axis=1).melt(var_name='Variable', value_name='Accuracy')
sns.boxplot(data=melted_df, x='Variable', y='Accuracy', hue='Variable', palette=palette,
            width=0.7, dodge=False, ax=ax)

ax.legend_.remove()  # remove the legend, as the information is already present in the x labels
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
ax.set_xlabel('', fontsize=20)  # remove unuseful xlabel ('Variable')
ax.set_ylabel("Inertia[s]", fontsize=20)
sns.despine(top=True, right=True, left=True, bottom=False)

plt.legend([
            '',
            'Inertia_S1',
            'Inertia_S2',
            'Inertia_S3',
            'Inertia_S4',
            'Inertia_S5',
            'Inertia_M0',
            'Inertia_M1',
            'Inertia_M2',
            'Inertia_M3',
            'Inertia_M4',
            'Inertia_M5', 
            ], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize = 22)
