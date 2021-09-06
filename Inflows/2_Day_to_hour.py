# -*- coding: utf,-8 -*-
"""
Created on Tue May 11 14:59:35 2021

@author: VLIR_energy
"""
import pandas as pd

data = pd.read_csv("ScaledInflows1_CE.csv",index_col=0)
data.index = pd.to_datetime(data.index)

new_index = pd.date_range(start="1980-01-01 00:00", end="2015-12-31 23:55",freq="1h")

data_newindex = data.reindex(index=new_index)

data_fill= data_newindex.fillna(method="pad")

data_fill.to_csv("ScaledInflows_CE.csv")
















