# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 16:32:38 2022

@author: Francisco Mena
"""
import sqlite3
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from calendar import monthrange
import holidays
from print_model_summary import print_model_summary

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import lightgbm as lgb
import joblib

#%%
class modeloCencosud:

    @staticmethod
    def collect_data(table_name):
        """
        Function to obtain the data from the database
        
        table_name: str
            Name of the SQL Table        
        """
        
        # if type(table_name!= str):
        #     sys.exit("table_name is not a string, please correct")
        
        
        # creating file path to database
        dbfile = os.path.join( os.getcwd(), "problemas_seleccion.db")
        # Create a SQL connection to our SQLite database
        con = sqlite3.connect(dbfile)

        # creating cursor
        cur = con.cursor()

        # reading all table names
        table_list = [a for a in cur.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]
        # here is you table list
        print(table_list)

        # Be sure to close the connection
        # con.close()

        #Download table
        sql_query = "SELECT * FROM " + table_name
        df = pd.read_sql(sql_query, con)

        return df

#%%
    @staticmethod
    def feature_engineering(df):
        """
        Function to generate the new features that will be used during training
        
        df: pandas dataframe
            Dataframe with the meat sales information
        """
        
        
        
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

        
        #Sort dataframe by date, and return columns that are relevant for training
        df.sort_values(by = "FECHA", inplace = True)

        return df.copy()
    
    
#%% 
    @staticmethod    
    def ExploratoryDataAnalysis(df):
        """
        Function to plot different features vs Meat Sales
        
        df: pandas dataframe
            Dataframe with the meat sales information
        """
        
        folder2save = os.path.join( os.getcwd(), "Plots")
        
        if not os.path.exists( os.path.join( os.getcwd(), "Plots")):            
            os.mkdir( os.path.join( os.getcwd(), "Plots") )
        
        #There are more sales on September and December
        sns.boxplot(data = df, x =  "month", y = "VENTAS")
        plt.savefig(folder2save + "/Boxplot_SalesPerMonth.png", dpi = 150, bbox_inches = "tight")        
        plt.close()

        #There are more sales on Friday, Saturday, and Sunday        
        sns.boxplot(data = df, x =  "day_name", y = "VENTAS", order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        plt.savefig(folder2save + "/Boxplot_SalesPerDay.png", dpi = 150, bbox_inches = "tight")        
        plt.close()

        #Slightly higher sales on holidays
        sns.boxplot(data = df, x =  "is_holiday", y = "VENTAS")
        plt.ylim(0,5000)
        plt.savefig(folder2save + "/Boxplot_SalesPerHoliday.png", dpi = 150, bbox_inches = "tight")        
        plt.close()

        #Tienda 1 sells the most, while 2 the least
        sns.boxplot(data = df, x =  "TIENDA", y = "VENTAS")
        plt.savefig(folder2save + "/Boxplot_SalesPerStore.png", dpi = 150, bbox_inches = "tight")        
        plt.close()

        #There are clear peaks in the data
        plt.figure(figsize = (12,6))
        sns.lineplot(data = df,x = "FECHA", y = "VENTAS")
        plt.savefig(folder2save + "/TimeSeries_Sales.png", dpi = 150, bbox_inches = "tight")        
        plt.close()

        #There are more sales on Sept and Dec
        df["year"] = df.FECHA.dt.year
        sns.boxplot(data = df, x = "year", y = "VENTAS")
        plt.savefig(folder2save + "/Boxplot_SalesPerMonth.png", dpi = 150, bbox_inches = "tight")
        plt.close()
       
        
#%%

    @staticmethod    
    def find_best_model(df):
        """
        Function that trains trains 3 different models: Random Forest, SVR, and ElasticNetCV

        Parameters
        ----------
        df : pandas DataFrame
            Dataframe with the meat sales information

        Returns
        -------
        Saves text files with the performance metrics of the 3 models,
        Saves the best model with joblib, including one-hot encoders and scaler,
        Returns dictionary with the best model, including one-hot encoders and scaler,

        """
        
        
        featcols = ['TIENDA', 'day_x', 'day_y', 'day_name', 'dayofweek_x', 'dayofweek_y', 'month_x',
        'month_y', 'weekend', 'is_holiday', 'VENTAS']

        df.sort_values(by = "FECHA", inplace = True)

        df2 = df[featcols].copy()


        ############# One Hot Encoder        
        df2["TIENDA"] = ["TIENDA_" + str(x) for x in df2["TIENDA"]]

        catvars = ["TIENDA", "day_name"]


        ohe_tienda = OneHotEncoder()
        A = pd.DataFrame(  ohe_tienda.fit_transform(df2[["TIENDA"]]).toarray(), columns = ohe_tienda.categories_[0] , index = df2.index)
        df2 = df2.merge(A[ohe_tienda.categories_[0]], left_index = True, right_index = True)

        ohe_day = OneHotEncoder()
        B = pd.DataFrame(  ohe_day.fit_transform(df2[["day_name"]]).toarray(), columns = ohe_day.categories_[0] , index = df2.index)
        df2 = df2.merge(B[ohe_day.categories_[0]], left_index = True, right_index = True)
            
        df2.drop(catvars, axis = 1, inplace = True)



        # SPLIT between training and testing
        
        y = df2.pop("VENTAS")
        X = df2.copy()


        #Ill use 2019 for testing, and previous years for training
        year = df.FECHA.dt.year
        idxtest = year.loc[year == 2019].index
        idxtrain = year.loc[year != 2019].index
        
        Xtrain = X.loc[idxtrain]
        ytrain = y.loc[idxtrain]
        Xtest = X.loc[idxtest]
        ytest = y.loc[idxtest]


        #Scale
        scale = StandardScaler()

        Xtrain_scaled = scale.fit_transform(Xtrain)
        Xtest_scaled = scale.transform(Xtest)


        """We'll test 3 models"""
        print("Training ElasticNet")
        reg_EN = ElasticNetCV(l1_ratio = [.1, .5, .7, .9, .95, .99, 1], cv = 3, n_jobs = -1, random_state=42)
        
        
        reg_EN.fit(Xtrain_scaled, ytrain)
        
        
        
        ypred = reg_EN.predict(Xtest_scaled)
        
        
        print("RMSE")
        print(mean_squared_error(ytest, ypred, squared=False))
        print("MAE")
        print(mean_absolute_error(ytest, ypred))
        
        relerr = np.abs(ypred - ytest)/ytest
        print("Mean Relative Error")
        print(np.mean(relerr))
        
        RMSE_EN = np.round( mean_squared_error(ytest, ypred, squared=False), 2)
        MAE_EN = np.round( mean_absolute_error(ytest, ypred) )
        print_model_summary("ElasticNetCV", reg_EN, RMSE_EN, MAE_EN)
        
        folder2save = os.path.join( os.getcwd(), "Plots")    
        plt.figure(figsize = (6,6))
        plt.scatter(ytest, ypred, marker = 'x', color = 'r')
        plt.plot(ytest, ytest, 'b')
        plt.xlabel('True value')
        plt.ylabel('Predicted value')
        plt.xlim(0,6000)
        plt.ylim(0,6000)
        plt.savefig(folder2save + "/Regression_ElasticNetCV.png", dpi = 150, bbox_inches = "tight")
        
        
        """ Random Forest """
        print("Training Random Forest")
        
        reg = RandomForestRegressor(random_state=42, criterion = "absolute_error")

        # Number of trees in random forest
        n_estimators = [10, 100, 200]
        # Number of features to consider at every split
        # max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [10, 20, None]
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        # Create the random grid
        params = {'n_estimators': n_estimators,
                       # 'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}


        grid = GridSearchCV(estimator = reg, param_grid = params, 
                            scoring = "neg_mean_absolute_error", 
                            cv = 3, n_jobs = 12, verbose = 2)


        grid.fit(Xtrain_scaled, ytrain)

        print(grid.best_params_)

        reg_RF = grid.best_estimator_


        ypred = reg_RF.predict(Xtest_scaled)


        print("RMSE")
        print(mean_squared_error(ytest, ypred, squared=False))
        print("MAE")
        print(mean_absolute_error(ytest, ypred))

        relerr = np.abs(ypred - ytest)/ytest
        print("Mean Relative Error")
        print(np.mean(relerr))


        RMSE_RF = np.round( mean_squared_error(ytest, ypred, squared=False), 2)
        MAE_RF = np.round( mean_absolute_error(ytest, ypred) )
        print_model_summary("RandomForest", reg_RF, RMSE_RF, MAE_RF)

        plt.figure(figsize = (6,6))
        plt.scatter(ytest, ypred, marker = 'x', color = 'r')
        plt.plot(ytest, ytest, 'b')
        plt.xlabel('True value')
        plt.ylabel('Predicted value')
        plt.xlim(0,6000)
        plt.ylim(0,6000)
        plt.savefig(folder2save + "/Regression_RandomForest.png", dpi = 150, bbox_inches = "tight")



        """ LightGBM """        
        
        reg_lgb = lgb.LGBMRegressor(random_state=42, objective="regression")
        
        reg_lgb.fit(Xtrain_scaled, ytrain)
        
        
        ypred = reg_lgb.predict(Xtest_scaled)
        
        
        print("RMSE")
        print(mean_squared_error(ytest, ypred, squared=False))
        print("MAE")
        print(mean_absolute_error(ytest, ypred))
        
        relerr = np.abs(ypred - ytest)/ytest
        print("Mean Relative Error")
        print(np.mean(relerr))
        
        RMSE_lgb = np.round( mean_squared_error(ytest, ypred, squared=False), 2)
        MAE_lgb = np.round( mean_absolute_error(ytest, ypred) )
        print_model_summary("LightGBM", reg_lgb, RMSE_lgb, MAE_lgb)
        
        plt.figure(figsize = (6,6))
        plt.scatter(ytest, ypred, marker = 'x', color = 'r')
        plt.plot(ytest, ytest, 'b')
        plt.xlabel('True value')
        plt.ylabel('Predicted value')
        plt.xlim(0,6000)
        plt.ylim(0,6000)
        plt.savefig(folder2save + "/Regression_LightGBM.png", dpi = 150, bbox_inches = "tight")        


        """ Return model with lowest MAE and save as joblib"""
        
        models = [reg_EN, reg_RF, reg_lgb]

        # RMSE = [RMSE_EN, RMSE_RF, RMSE_lgb]
        MAE = [MAE_EN, MAE_RF, MAE_lgb]
        
        # best_model = models[ np.argmin(RMSE) ]
        best_model = models[ np.argmin(MAE) ]

        print('\n')
        print("Best model is " + str(best_model) )
        
        dictio_model = {'ohe_day': ohe_day,
                        'ohe_tienda': ohe_tienda,
                        'scaler': scale,
                        'model': best_model
                        }
        
        joblib.dump(dictio_model, 'modelo_carne_Cencosud.joblib')
        
        
        return dictio_model
        
        
        
#%%        
    @staticmethod
    def predict_meat_sales(date):
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
        df = modeloCencosud().feature_engineering(df)
        
        
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
        
        return y
        
#%%
        