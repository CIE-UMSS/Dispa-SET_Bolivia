# -*- coding: utf-8 -*-
"""
Created on Sat May 15 07:12:57 2021

@author: VLIR_energy
"""

import pandas as pd

for z in ['OR','CE','NO','SU']:
    df1 = pd.read_csv("Inflows_"+z+".csv", index_col=0)
    
    df2 = pd.read_csv("MainData_SIN.csv", index_col=0)
    # To Convert DataFrame to Series
    df2_2 = df2.squeeze()
    
    # ScaledInflows= P / P_diseño
    # P= Q*h*g*ρ/1E6
    df3 = (df1*df2_2["Falls down (M)"]*9.81*1019/1E6/df2_2["Power (MW)"])
    df4 = df3.dropna(1)
    
    # Converting the index as data
    df4.index = pd.to_datetime(df4.index)
    
    new_index = pd.date_range(start="1980-01-01 00:00", end="2015-12-31 23:55",freq="1h")
    df4 = df4.reindex(index=new_index)
    data= df4.fillna(method="pad")
    
    for i in range(1980,2016):
        startdate = "01-01-"+str(i)+" 00:00" 
        endate = "12-31-"+str(i)+" 23:55"  
        idx = pd.date_range(start=startdate, end=endate,freq="1h") 
        df1  = data.loc[idx,:]        
        filename="../HydroData/ScaledInflows/"+z+"/" + str(i) + ".csv"  
        print(filename)
        df1.to_csv(filename)
    
    

    


    
    


    
  