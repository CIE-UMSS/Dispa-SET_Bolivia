# -*- coding: utf-8 -*-
"""
Created on Mon May 10 15:23:12 2021

@author: VLIR_energy
"""

import pandas as pd


data = pd.read_excel("data.xlsx", index_col=0)

new_index= pd.date_range(start="01-01-1980 00:00", end="12-31-2015 23:55", freq="1h")

data_add = data.reindex(index=new_index)

data_fill = data_add.fillna(method="pad")

data_fill.to_csv("Inflows_NO.csv")
data_fill.to_excel("Inflows_NO.xlsx")
