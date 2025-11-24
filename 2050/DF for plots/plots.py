# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 11:01:21 2023

@author: navia
"""
#############################################   1.PLOT DE INERCIAS SUPERPUESTAS
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
plt.style.use('default')
print(plt.style.available)
%matplotlib inline

df = pd.read_csv('INERTIA.csv',index_col=(0))
# Filter dates
# df.reset_index(level =0, inplace = True)
# mask1 = (df['TIMESTAMP'] > '2026-09-01 00:00:00+00:00') & (df['TIMESTAMP'] <= '2026-09-07 23:00:00+00:00')
# df = df.loc[mask1]
# df.set_index('TIMESTAMP',inplace=True, drop=True)
# code
# Visualizing The Open Price of all the stocks
  
# to set the plot size
plt.figure(figsize=(16, 8), dpi=150)
  
# using plot method to plot open prices.
# in plot method we set the label and color of the curve.
df['Inertia_S0'].plot(label='Inertia_S0', alpha=0.5, color='teal')
df['Inertia_S1'].plot(label='Inertia_S1', alpha=0.5, color='darkslategrey')
df['Inertia_S2'].plot(label='Inertia_S2', alpha=0.5, color='navy')
df['Inertia_S3'].plot(label='Inertia_S3', alpha=0.5, color='seagreen')
df['Inertia_S4'].plot(label='Inertia_S4', alpha=0.5, color='darkkhaki')
df['Inertia_S5'].plot(label='Inertia_S5', alpha=0.5, color='plum')
df['Inertia_M0'].plot(label='Inertia_M0', alpha=0.5, color='teal')
df['Inertia_M1'].plot(label='Inertia_M1', alpha=0.5, color='darkslategrey')
df['Inertia_M2'].plot(label='Inertia_M2', alpha=0.5, color='navy')
df['Inertia_M3'].plot(label='Inertia_M3', alpha=0.5, color='seagreen')
df['Inertia_M4'].plot(label='Inertia_M4', alpha=0.5, color='darkkhaki')
df['Inertia_M5'].plot(label='Inertia_M5', alpha=0.5, color='plum')
df['Mlimit'].plot(label='Mlimit', alpha=1, color='red')

# adding title to the plot
plt.title('System Inertia Hystogram', fontsize = 18)
  
# adding Label to the x-axis
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.xlabel('Time', fontsize = 15)
plt.ylabel('Inertia(H)[s]', fontsize = 15)
  
# adding legend to the curve
plt.legend()

#%%
############################################   2.PLOT DE DISTRIBUCION ACUMULADA
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

plt.style.use('default')
print(plt.style.available)
%matplotlib inline

#No. of data points used
N = 8761

df = pd.read_csv('INERTIA.csv',index_col=(0))
plt.figure(figsize=(32, 8), dpi=150, edgecolor='yellow')

#sort data in ascending order
# Inertia_S0 = np.sort(df['Inertia_S0'])
# Inertia_S1 = np.sort(df['Inertia_S1'])
# Inertia_S2 = np.sort(df['Inertia_S2'])
# Inertia_S3 = np.sort(df['Inertia_S3'])
# Inertia_S4 = np.sort(df['Inertia_S4'])
# Inertia_S5 = np.sort(df['Inertia_S5'])

# Inertia_M0 = np.sort(df['Inertia_M0'])
# Inertia_M1 = np.sort(df['Inertia_M1'])
# Inertia_M2 = np.sort(df['Inertia_M2'])
# Inertia_M3 = np.sort(df['Inertia_M3'])
# Inertia_M4 = np.sort(df['Inertia_M4'])
Inertia_M5 = np.sort(df['Inertia_M5'])

Mlimit = np.sort(df['Mlimit'])

# get the cdf values of y
y = np.arange(N) / float(N)

# adding title to the plot
plt.title('Cumulative Distribution of System Inertia', fontsize = 30)  

# plotting
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.xlabel('Inertia(H)[s]', fontsize = 25)
plt.ylabel('Distribution', fontsize = 25)
  
  
# plt.plot(Inertia_S0, y, marker='o', alpha=0.1, color='teal')
# plt.plot(Inertia_S1, y, marker='o', alpha=0.1, color='darkslategrey')
# plt.plot(Inertia_S2, y, marker='o', alpha=0.1, color='navy')
# plt.plot(Inertia_S3, y, marker='o', alpha=0.1, color='seagreen')
# plt.plot(Inertia_S4, y, marker='o', alpha=0.1, color='darkkhaki')
# plt.plot(Inertia_S5, y, marker='o', alpha=0.1, color='plum')

# plt.plot(Inertia_M0, y, marker='x', alpha=0.1, color='teal')
# plt.plot(Inertia_M1, y, marker='x', alpha=0.1, color='darkslategrey')
# plt.plot(Inertia_M2, y, marker='x', alpha=0.1, color='navy')
# plt.plot(Inertia_M3, y, marker='x', alpha=0.1, color='seagreen')
# plt.plot(Inertia_M4, y, marker='x', alpha=0.1, color='darkkhaki')
plt.plot(Inertia_M5, y, marker='x', alpha=0.1, color='plum')

plt.plot(Mlimit, y, marker='o', alpha=1, color='red')
# adding legend to the curve
plt.legend([
            # 'Inertia_S0',
            # 'Inertia_S1',
            # 'Inertia_S2',
            # 'Inertia_S3',
            # 'Inertia_S4',
            # 'Inertia_S5',
            # 'Inertia_M0',
            # 'Inertia_M1',
            # 'Inertia_M2',
            # 'Inertia_M3',
            # 'Inertia_M4',
            'Inertia_M5', 
            'Mlimit'
            ], fontsize = 25)
#%%
####################################################################################################
#no lo use porque no se notan bien las escalas:
# import pandas as pd
# from matplotlib import pyplot as plt
# import seaborn as sns
# plt.style.use('default')
# %matplotlib inline

# df = pd.read_csv('SCENARIO_5.csv',index_col=(0))
# # Filter dates
# # df.reset_index(level =0, inplace = True)
# # mask1 = (df['TIMESTAMP'] > '2026-09-01 00:00:00+00:00') & (df['TIMESTAMP'] <= '2026-09-07 23:00:00+00:00')
# # df = df.loc[mask1]
# # df.set_index('TIMESTAMP',inplace=True, drop=True)
# # code
# # Visualizing The Open Price of all the stocks
  
# # to set the plot size
# plt.figure(figsize=(32, 8), dpi=150)
# fig, ax = plt.subplots() 
# # using plot method to plot open prices.
# # in plot method we set the label and color of the curve.
# ax.plot(df['Inertia_S5'], label='Inertia_S5', alpha=0.3, color='darkcyan')
# ax.plot(df['Inertia_M5'], label='Inertia_M5', alpha=1, color='indigo')
# ax.plot(df['Mlimit'], label='Mlimit', color='red')

# ax1 = ax.twinx()

# ax1.plot(df['Curtailment_S5'], label='Curtailment_S5', alpha=0.1, color='blue')
 
# # adding title to the plot
# plt.title('System Inertia Hystogram')
  
# # adding Label to the x-axis
# plt.xlabel('Time')
# plt.ylabel('Inertia(H)[s]')
  
# # adding legend to the curve
# plt.legend()
#%%
#Plotting Penetration level histogram with pyplot
# importing Libraries
  
# import pandas as pd
import pandas as pd
  
# importing matplotlib module
import matplotlib.pyplot as plt
plt.style.use('default')
  
# %matplotlib inline: only draw static
# images in the notebook
%matplotlib inline
# importing the data
df0 = pd.read_csv('SCENARIO_0.csv',index_col=(0))
df1 = pd.read_csv('SCENARIO_1.csv',index_col=(0))
df2 = pd.read_csv('SCENARIO_2.csv',index_col=(0))
df3 = pd.read_csv('SCENARIO_3.csv',index_col=(0))
df4 = pd.read_csv('SCENARIO_4.csv',index_col=(0))
df5 = pd.read_csv('SCENARIO_5.csv',index_col=(0))
df0['Week'] = pd.DatetimeIndex(df.index).week
df1['Week'] = pd.DatetimeIndex(df.index).week
df2['Week'] = pd.DatetimeIndex(df.index).week
df3['Week'] = pd.DatetimeIndex(df.index).week
df4['Week'] = pd.DatetimeIndex(df.index).week
df5['Week'] = pd.DatetimeIndex(df.index).week
# to set the plot size
plt.figure(figsize=(16, 8), dpi=150)
  
# using plot method to plot open prices.
# in plot method we set the label and color of the curve.
# df0['PL_SG0'].plot(label='Penetration Level Scenario S0', color='teal',alpha=0.4), color='teal',alpha=0.4, color='darkslategrey',alpha=0.4, color='navy',alpha=0.4, color='seagreen',alpha=0.4, color='darkkhaki',alpha=0.4, color='plum',alpha=0.4
# df1['PL_SG1'].plot(label='Penetration Level Scenario S1', color='darkslategrey',alpha=0.4)
# df2['PL_SG2'].plot(label='Penetration Level Scenario S2', color='navy',alpha=0.4)
# df3['PL_SG3'].plot(label='Penetration Level Scenario S3', color='seagreen',alpha=0.4)
# df4['PL_SG4'].plot(label='Penetration Level Scenario S4', color='darkkhaki',alpha=0.4)
# df5['PL_S5'].plot(label='Penetration Level Scenario S5', color='plum',alpha=0.4)
# df['PL_M5'].plot(label='Penetration Level Scenario M5', color='blue',alpha=0.4)

plt.plot(df0['Week'],df0['PL_SG0'],label='Penetration Level Scenario S0', color='teal',alpha=1)
plt.plot(df1['Week'],df1['PL_SG1'],label='Penetration Level Scenario S1', color='darkslategrey',alpha=1)
plt.plot(df2['Week'],df2['PL_SG2'],label='Penetration Level Scenario S2', color='navy',alpha=1)
plt.plot(df3['Week'],df3['PL_SG3'],label='Penetration Level Scenario S3', color='seagreen',alpha=1)
plt.plot(df4['Week'],df4['PL_SG4'],label='Penetration Level Scenario S4', color='darkkhaki',alpha=1)
plt.plot(df5['Week'],df5['PL_S5'],label='Penetration Level Scenario S5', color='plum',alpha=1)


  
# adding title to the plot
plt.title('Histogram of Penetration Level of VRE per scenario')
  
# adding Label to the x-axis
plt.xlabel('TimeStep')
  
# adding legend to the curve
plt.legend()

#%%
#Plotting Penetration level histogram with seaborn
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
plt.style.use('default')
print(plt.style.available)
%matplotlib inline

df0 = pd.read_csv('SCENARIO_0.csv')
df1 = pd.read_csv('SCENARIO_1.csv')
df2 = pd.read_csv('SCENARIO_2.csv')
df3 = pd.read_csv('SCENARIO_3.csv')
df4 = pd.read_csv('SCENARIO_4.csv')
df5 = pd.read_csv('SCENARIO_5.csv')

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
# list_to_plot = list(df.columns)
# for i in list_to_plot:
fig, ax = plt.subplots(figsize=(20,7))
# sns.lineplot(data = df0, y='PL_SG0', x = 'Week' , linewidth=.8, color='teal',alpha=0.4,label='Penetration_Level_S0')
# sns.lineplot(data = df1, y='PL_SG1', x = 'Week' , linewidth=.8, color='darkslategrey',alpha=0.4,label='Penetration_Level_S1')
# sns.lineplot(data = df2, y='PL_SG2', x = 'Week' , linewidth=.8, color='navy',alpha=0.4,label='Penetration_Level_S2')
# sns.lineplot(data = df3, y='PL_SG3', x = 'Week' , linewidth=.8, color='seagreen',alpha=0.4,label='Penetration_Level_S3')
# sns.lineplot(data = df4, y='PL_SG4', x = 'Week' , linewidth=.8, color='darkkhaki',alpha=0.4,label='Penetration_Level_S4')
sns.lineplot(data = df5, y='PL_SG5', x = 'Hour' , linewidth=.8, color='black',alpha=0.2,label='Penetration_Level_S5')     
sns.lineplot(data = df5, y='PL_MG5', x = 'Hour' , linewidth=.8, color='plum',alpha=1,label='Penetration_Level_M5') 
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)
plt.xlabel('Week', fontsize = 12)
plt.ylabel('Penetration Level[pu]', fontsize = 12)
plt.show()

#%%
#Plotting Penetration level histogram with seaborn
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
plt.style.use('default')
print(plt.style.available)
%matplotlib inline
# 1. System Inertia between with and without inertia
plt.figure(figsize=(15,7))
sns.lineplot(data = df0, y='Inertia_S0', x = 'TIMESTAMP' , linewidth=1, color='black',alpha=0.2,label='Inertia_S0',n_boot=100)
sns.lineplot(data = df0, y='Inertia_M0', x = 'TIMESTAMP' , linewidth=1, color='teal',alpha=1,label='Inertia_M0',n_boot=100)
sns.lineplot(data = df0, y='Mlimit', x = 'TIMESTAMP' , linewidth=1, color='red',alpha=1,label='Inertia_Limit',n_boot=100)      
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize = 12)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)
plt.xlabel('TIMESTAMP', fontsize = 12)
plt.ylabel('Inertia[s]', fontsize = 12)
plt.show()

plt.figure(figsize=(15,7))
sns.lineplot(data = df1, y='Inertia_S1', x = 'TIMESTAMP' , linewidth=1, color='black',alpha=0.2,label='Inertia_S1',n_boot=100)
sns.lineplot(data = df1, y='Inertia_M1', x = 'TIMESTAMP' , linewidth=1, color='darkslategrey',alpha=1,label='Inertia_M1',n_boot=100)
sns.lineplot(data = df1, y='Mlimit', x = 'TIMESTAMP' , linewidth=1, color='red',alpha=1,label='Inertia_Limit',n_boot=100)      
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize = 12)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)
plt.xlabel('TIMESTAMP', fontsize = 12)
plt.ylabel('Inertia[s]', fontsize = 12)
plt.show()

plt.figure(figsize=(15,7))
sns.lineplot(data = df2, y='Inertia_S2', x = 'TIMESTAMP' , linewidth=1, color='black',alpha=0.2,label='Inertia_S2',n_boot=100)
sns.lineplot(data = df2, y='Inertia_M2', x = 'TIMESTAMP' , linewidth=1, color='navy',alpha=1,label='Inertia_M2',n_boot=100)
sns.lineplot(data = df2, y='Mlimit', x = 'TIMESTAMP' , linewidth=1, color='red',alpha=1,label='Inertia_Limit',n_boot=100)      
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize = 12)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)
plt.xlabel('TIMESTAMP', fontsize = 12)
plt.ylabel('Inertia[s]', fontsize = 12)
plt.show()

plt.figure(figsize=(15,7))
sns.lineplot(data = df3, y='Inertia_S3', x = 'TIMESTAMP' , linewidth=1, color='black',alpha=0.2,label='Inertia_S3',n_boot=100)
sns.lineplot(data = df3, y='Inertia_M3', x = 'TIMESTAMP' , linewidth=1, color='seagreen',alpha=1,label='Inertia_M3',n_boot=100)
sns.lineplot(data = df3, y='Mlimit', x = 'TIMESTAMP' , linewidth=1, color='red',alpha=1,label='Inertia_Limit',n_boot=100)      
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize = 12)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)
plt.xlabel('TIMESTAMP', fontsize = 12)
plt.ylabel('Inertia[s]', fontsize = 12)
plt.show()

plt.figure(figsize=(15,7))
sns.lineplot(data = df4, y='Inertia_S4', x = 'TIMESTAMP' , linewidth=1, color='black',alpha=0.2,label='Inertia_S4',n_boot=100)
sns.lineplot(data = df4, y='Inertia_M4', x = 'TIMESTAMP' , linewidth=1, color='darkkhaki',alpha=1,label='Inertia_S4',n_boot=100)
sns.lineplot(data = df4, y='Mlimit', x = 'TIMESTAMP' , linewidth=1, color='red',alpha=1,label='Inertia_Limit',n_boot=100)      
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize = 12)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)
plt.xlabel('TIMESTAMP', fontsize = 12)
plt.ylabel('Inertia[s]', fontsize = 12)
plt.show()

plt.figure(figsize=(15,7))
sns.lineplot(data = df5, y='Inertia_S5', x = 'TIMESTAMP' , linewidth=1, color='black',alpha=0.2,label='Inertia_S5',n_boot=100)
sns.lineplot(data = df5, y='Inertia_M5', x = 'TIMESTAMP' , linewidth=1, color='plum',alpha=1,label='Inertia_M5',n_boot=100)
sns.lineplot(data = df5, y='Mlimit', x = 'TIMESTAMP' , linewidth=1, color='red',alpha=1,label='Inertia_Limit',n_boot=100)      
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize = 12)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)
plt.xlabel('TIMESTAMP', fontsize = 12)
plt.ylabel('Inertia[s]', fontsize = 12)
plt.show()


#%%
####################################################################################################

# sns.kdeplot(scenario5.PL_S5, shade=True,color='green');
# sns.kdeplot(scenario5.Curtailment_S5, shade=True,color='red');
# # sns.kdeplot(scenario5.PL_S5, cumulative=True);


# # sns.kdeplot(scenario5.Inertia_SG5, scenario5.Curtailment_SG5);


# sns.set_style('darkgrid')
# sns.ecdfplot(x='Inertia_S5', data=scenario5,c='b')
# sns.ecdfplot(x='Inertia_M5', data=scenario5,c='g')
# plt.axvline(7.14, c='red')



# sns.catplot(y='Inertia_MG5', data=scenario5);
# sns.catplot(y='Inertia_SG5', data=scenario5);

# sns.catplot(x=('Inertia_MG5','Inertia_SG5'), data=scenario5, kind='box');
# sns.catplot(x='Inertia_SG5', data=scenario5, kind='box');


# df = pd.read_csv('SCENARIO_5.csv')
# df['TIMESTAMP']=df['TIMESTAMP'].astype('datetime64[s]')
# df.head()
# df['Day']=df.TIMESTAMP.dt.date
# df['Month']=df.TIMESTAMP.dt.month
# df['Hour']=df.TIMESTAMP.dt.hour

# sns.set_style('dark')
# sns.lineplot(x=df.TIMESTAMP, y=df.PL_M5)


# df = pd.read_csv('SCENARIO_5.csv',index_col=(0))
# dfi = pd.DataFrame()
# dfi = df.drop(['Curtailment_S5', 'Curtailment_M5', 'ShedLoad_S5', 'ShedLoad_M5', 'OutputSystemCost_S5', 'OutputSystemCost_M5', 'ShadowPrice_S5', 'ShadowPrice_M5'], axis=1)
# dfi.reset_index(level =0, inplace = True)
# dfi['Day']=dfi.TIMESTAMP.dt.date
# mask1 = (dfi['TIMESTAMP'] > '2026-09-01 00:00:00+00:00') & (dfi['TIMESTAMP'] <= '2026-09-07 23:00:00+00:00')
# dfi = dfi.loc[mask1]
# dfi.set_index('TIMESTAMP',inplace=True, drop=True)
# plt.xticks(rotation=90)
# dfi.plot(kind='line', color=('red','b','g'), xticks=[], rot=0, fontsize=8)



# df = pd.read_csv('SCENARIO_5.csv',index_col=(0))
# dfi = pd.DataFrame()
# dfi = df.drop(['Mlimit', 'Inertia_S5', 'Inertia_M5','Curtailment_M5', 'ShedLoad_S5', 'ShedLoad_M5', 'OutputSystemCost_S5', 'OutputSystemCost_M5', 'ShadowPrice_S5', 'ShadowPrice_M5','TRG_S5','TRG_M5','TG_S5','TG_M5','PL_S5','PL_M5'], axis=1)
# dfi.reset_index(level =0, inplace = True)
# dfi['Day']=dfi.TIMESTAMP.dt.date
# mask1 = (dfi['TIMESTAMP'] > '2026-09-01 00:00:00+00:00') & (dfi['TIMESTAMP'] <= '2026-09-07 23:00:00+00:00')
# dfi = dfi.loc[mask1]
# dfi.set_index('TIMESTAMP',inplace=True, drop=True)
# plt.xticks(rotation=90)
# dfi.plot(kind='line', color=('r','g'), rot=30, fontsize=8)

# df = pd.read_csv('SCENARIO_5.csv',index_col=(0))
# dfi = pd.DataFrame()
# dfi = df.drop(['Mlimit', 'Inertia_S5', 'Inertia_M5','Curtailment_S5','Curtailment_M5', 'ShedLoad_S5', 'ShedLoad_M5', 'OutputSystemCost_S5', 'OutputSystemCost_M5', 'ShadowPrice_S5', 'ShadowPrice_M5','TRG_S5','TRG_M5','TG_S5','TG_M5','PL_M5'], axis=1)
# dfi.reset_index(level =0, inplace = True)
# dfi['Day']=dfi.TIMESTAMP.dt.date
# mask1 = (dfi['TIMESTAMP'] > '2026-09-01 00:00:00+00:00') & (dfi['TIMESTAMP'] <= '2026-09-07 23:00:00+00:00')
# dfi = dfi.loc[mask1]
# dfi.set_index('TIMESTAMP',inplace=True, drop=True)
# plt.xticks(rotation=90)
# dfi.plot(kind='line', color=('g'), rot=30, fontsize=8)



# plt.style.use('bmh')
# df = pd.read_csv('infobyzones.csv')
# x = df['Zones']
# y = df['Demand']
# plt.xlabel('Zones', fontsize=0.8) 
# plt.ylabel('Demand', fontsize=0.8) 
# plt.bar(x,y)
# plt.show()


# import pandas as pd
# import matplotlib.pyplot as plt

# s = pd.Series([2.58, 4.09, 2.53, 2.24], index = ["CE", "OR", "NO", "SU"])

# #Set descriptions:
# plt.title("Total Power Demand by Zone")
# plt.ylabel('Demand [MW]')
# plt.xlabel('Zones')

# #Set tick colors:
# ax = plt.gca()
# ax.tick_params(axis='x', colors='blue')
# ax.tick_params(axis='y', colors='red')

# #Plot the data:
# my_colors = 'rgbkymc'  #red, green, blue, black, etc.

# pd.Series.plot(s, kind='bar', color=my_colors,)

# plt.show()
