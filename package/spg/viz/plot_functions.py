import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.offline as pyo
import pandas as pd

from bayes_opt import BayesianOptimization
import shap
import wandb
import xgboost
from xgboost import cv, XGBRegressor

# Set notebook mode to work in offline
pyo.init_notebook_mode()


def plot_residuals(y_true, y_pred):
    '''Plot residuals of the model
       @param y_true: true values
       @param y_pred: predicted values
    '''
    plt.figure(figsize=(10, 5))
    plt.scatter(y_true, y_pred - y_true)
    plt.hlines(y=0, xmin=y_true.min(), xmax=y_true.max(), color='red')
    plt.xlabel('y_true')
    plt.ylabel('y_pred - y_true')
    plt.title('Residuals')
    plt.show();

def plot_distribution(y_true, y_pred):
    '''Plot distribution of the model
        @param y_true: true values
        @param y_pred: predicted values
    '''
    sns.distplot(y_true, color='r', label='real production');
    sns.distplot(y_pred, color='b', label='predicted production');
    plt.legend();

def plot_feature_importance(model, X):
    '''Plot feature importance of the model
        @param model: model
        @param X: features
    '''    
    plt.figure(figsize=(10, 20))
    sorted_idx = model.feature_importances_.argsort()
    plt.barh(X.columns[sorted_idx], model.feature_importances_[sorted_idx]);
    plt.xlabel("Xgboost Feature Importance");

#Plotting scatter plot of predicted vs actual values
def plot_scatter_real_vs_predicted(y_true, y_pred):
    '''Plot scatter plot of predicted vs actual values
        @param y_true: true values
        @param y_pred: predicted values
    '''
    plt.scatter(y_true, y_pred)
    plt.xlabel('True Values [Production]')
    plt.ylabel('Predictions [Production]')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0,plt.xlim()[1]])
    plt.ylim([0,plt.ylim()[1]])
    _ = plt.plot([-100, 100], [-100, 100])

def plot_error_distribution(y_true, y_pred):
    '''Plot error distribution of the model
         @param y_true: true values
         @param y_pred: predicted values
    '''
    error = y_pred - y_true
    plt.hist(error, bins = 25)
    plt.xlabel("Prediction Error [Production]")
    _ = plt.ylabel("Count")