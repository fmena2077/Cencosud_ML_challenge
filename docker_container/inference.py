# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 19:32:29 2022

@author: franc
"""
import os
import pandas as pd
import numpy as np
import joblib
from calendar import monthrange
import holidays

#%%

def lets_predict_meat_sales(date):
    """
    Function that, given a date, predicts the meat sales in two more weeks

    Parameters
    ----------
    date : str
        Current date as a string

    Returns
    -------
    The meat sales prediction

    """
    # date = '2019-12-13'
    
    #Predict for two more weeks
    date2predict = pd.to_datetime(date) + pd.Timedelta(weeks=2)
    
    #We need predictions for each store
    df = pd.DataFrame({'FECHA': [date2predict]*3,
                       'TIENDA': [1, 2, 3]})


    """ Generate features """
    #day of the week, numeric
    df["dayofweek"] = df["FECHA"].dt.dayofweek

    #day of the week, string
    df["day_name"] = df["FECHA"].dt.day_name()

    #EDA shows that meat sales are higher on Friday and weekends
    df["weekend"] = df["day_name"].isin(["Friday", "Saturday", "Sunday"])


    #Convert day of the week feature to cyclical feature with period of 7 days
    df["dayofweek_x"] = np.sin( 2 * np.pi * df["dayofweek"] / 7 )
    df["dayofweek_y"] = np.cos( 2 * np.pi * df["dayofweek"] / 7 )

    #Month, numeric
    df["month"] = df["FECHA"].dt.month
    
    #Convert month feature to cyclical feature with period of 12 months
    df["month_x"] = np.sin( 2 * np.pi * df["month"] / 12 )
    df["month_y"] = np.cos( 2 * np.pi * df["month"] / 12 )

    
    #Day, numeric
    df["day"] = df["FECHA"].dt.day
    
    #We find the number of days per month to properly find the period of the cycle
    num_days_per_month = [monthrange(Y, M)[1] for Y,M in zip(df["FECHA"].dt.year, df["FECHA"].dt.month )]
    df["day_x"] = np.sin( 2 * np.pi * df["day"] / num_days_per_month )
    df["day_y"] = np.cos( 2 * np.pi * df["day"] / num_days_per_month )

    
    #EDA shows that meat sales are higher on holidays
    CL_holidays = holidays.CountryHoliday('Chile')
    df["FECHA"].isin( CL_holidays ).sum()
    df["is_holiday"] = [x in CL_holidays for x in df["FECHA"]]


    
    """ load model and predict """                
    featcols = ['TIENDA', 'day_x', 'day_y', 'day_name', 'dayofweek_x', 'dayofweek_y', 'month_x',
    'month_y', 'weekend', 'is_holiday']

    df2 = df[featcols].copy()

    df2["TIENDA"] = ["TIENDA_" + str(x) for x in df2["TIENDA"]]
    
    ############# One Hot Encoder        
    catvars = ["TIENDA", "day_name"]

    dictio_model = joblib.load( os.path.join(os.getcwd(), "modelo_carne_Cencosud.joblib"))

    ohe_day = dictio_model["ohe_day"]
    ohe_tienda = dictio_model["ohe_tienda"]
    scale = dictio_model["scaler"]
    model = dictio_model["model"]
    
    A = pd.DataFrame(  ohe_tienda.transform(df2[["TIENDA"]]).toarray(), columns = ohe_tienda.categories_[0] , index = df2.index)
    df2 = df2.merge(A[ohe_tienda.categories_[0]], left_index = True, right_index = True)
    
    B = pd.DataFrame(  ohe_day.transform(df2[["day_name"]]).toarray(), columns = ohe_day.categories_[0] , index = df2.index)
    df2 = df2.merge(B[ohe_day.categories_[0]], left_index = True, right_index = True)

    df2.drop(catvars, axis = 1, inplace = True)


    ############## Scale
    X = scale.transform(df2)

    y = model.predict(X)
    
    print("Using current's date " + date + ", the meat sales in two more weeks will be:\n")
    print("TIENDA 1: " + str(np.round(y[0],1)) +"\n")
    print("TIENDA 2: " + str(np.round(y[1],1)) +"\n")
    print("TIENDA 3: " + str(np.round(y[2],1)) +"\n")

    f = open(  os.path.join(os.getcwd(),  'prediction_meat_sales.txt'), 'w')
    f.write("DATE: " + date +"\n")
    f.write("TIENDA 1: " + str(np.round(y[0],1)) +"\n")
    f.write("TIENDA 2: " + str(np.round(y[1],1)) +"\n")
    f.write("TIENDA 3: " + str(np.round(y[2],1)) +"\n")
    f.close()
    
    return y

#%%

# date = str(input())
DATE = os.environ["DATE"]
lets_predict_meat_sales(DATE)

lets_predict_meat_sales("2019-12-13")
