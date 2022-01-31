# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 16:48:44 2022

@author: franc
"""

import pandas as pd

import os
os.chdir("C:\\Users\\franc\\projects\\letstf2gpu\\Cencosud")

from clase_Cencosud import modeloCencosud 

#%%

""" 
1st. Collect data
"""

cnc = modeloCencosud()


dfventas = cnc.collect_data('TABLA_VENTAS')

dfcat = cnc.collect_data("TABLA_CATEGORIAS")


"""
Select only "Carne" since it's the product to analyze
"""

cat_carne = dfcat.loc[dfcat["CATEGORIA"]=="CARNES", "ID_CATEGORIA"].values[0]
dfventas = dfventas.loc[ dfventas["ID_CATEGORIA"] == cat_carne ]


#FECHA to datetime 
dfventas["FECHA"] = pd.to_datetime(dfventas["FECHA"])


"""
2nd. Create new features for training
"""

df = cnc.feature_engineering(dfventas)
df.columns


"""
3rd. Visualizations of the features (EDA) - optional
"""

cnc.ExploratoryDataAnalysis(df)



"""
4th. Find the best machine learning model that predicts meat sales
    and export it as a dictionary
"""

cnc.find_best_model(df)



"""
5th. Function that predicts the meat sales in two more weeks
"""
date_today = '2019-12-13'
meat_sales = cnc.predict_meat_sales(date_today)

