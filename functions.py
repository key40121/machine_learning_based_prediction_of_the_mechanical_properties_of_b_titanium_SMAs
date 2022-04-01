import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


def model_evaluation(model, X_train, X_test, y_train, y_test):
    
    model.fit(X_train, y_train)
    predictions_test = model.predict(X_test)
    predictions_train = model.predict(X_train)
    
    r2_test = metrics.r2_score(y_test, predictions_test)
    r2_train = metrics.r2_score(y_train, predictions_train)
    
    MAE_test = metrics.mean_absolute_error(y_test, predictions_test)
    MSE_test = metrics.mean_squared_error(y_test, predictions_test)
    RMSE_test = np.sqrt(metrics.mean_squared_error(y_test, predictions_test))
    
    
    return r2_train, r2_test, MAE_test, MSE_test, RMSE_test