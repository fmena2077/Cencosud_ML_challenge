# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 16:27:55 2022

@author: franc
"""

import os

def print_model_summary(nombre_modelo, params, RMSE, MAE):
    """    
    Funcion para los resultados del modelo de scikit-learn en un texto.
    Parameters
    ----------
    nombre_modelo : nombre que uno le da al modelo
    params: parametros del modelo
    classificationReport : classification report de sklearn
    confusionMatrix : confusion matrix de sklearn    
    -------
    """    
    
    folder2save = os.path.join( os.getcwd(), "Model_metrics")
    
    if not os.path.exists( folder2save ):
        os.mkdir( os.path.join( os.getcwd(), "Model_metrics") )
    
    f = open(  os.path.join(folder2save,  nombre_modelo+ '_res.txt'), 'w')
    f.write('Name: ' + nombre_modelo + '\n')
    f.write('Model: ' + str(params)+ '\n')
    
    f.write('RMSE:'+ '\n')
    f.write(str( RMSE) + '\n')
    f.write('MAE:'+ '\n')
    f.write( str( MAE) + '\n')
    f.write('\n')
    f.close()