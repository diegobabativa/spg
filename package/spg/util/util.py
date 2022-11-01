import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def metrics_to_df(model_name, y_true, y_pred):
    '''Return metrics of the model in a dataframe
        @param y_true: true values
        @param y_pred: predicted values
    '''
    return pd.DataFrame({ 
                         'model': model_name,        
                         'MAE': [mean_absolute_error(y_true, y_pred)],
                         'MSE': [mean_squared_error(y_true, y_pred)],
                         'RMSE': [np.sqrt(mean_squared_error(y_true, y_pred))],
                         'RMSLE': [np.log(np.sqrt(mean_squared_error(y_true,y_pred)))],
                         'R2': [r2_score(y_true, y_pred)]})


def print_metrics(y_true, y_pred):
    '''Print metrics of the model
        @param y_true: true values
        @param y_pred: predicted values
    '''
    print("MAE:  ", mean_absolute_error(y_true, y_pred))
    print("MSE : ", mean_squared_error(y_true, y_pred))
    print("RMSE: ", np.sqrt(mean_squared_error(y_true, y_pred)))
    print("RMSLE:", np.log(np.sqrt(mean_squared_error(y_true,y_pred))))
    print("R2:   ", r2_score(y_true, y_pred))