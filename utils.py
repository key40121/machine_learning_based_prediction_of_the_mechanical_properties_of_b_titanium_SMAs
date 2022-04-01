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

#     print('r2 test:', r2_score(y_test, predictions_test))
#     print('r2 train:',r2_score(y_train, predictions_train))

#     print('MAE:', metrics.mean_absolute_error(y_test, predictions_test))
#     print('MSE:', metrics.mean_squared_error(y_test, predictions_test))
#     print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions_test)))
    
    
def plot_UTS(model, df, save_fig=False, file_name="sample.png"):

    df_y = df[['UTS']]
    df_X = df[['Ti', 'Mo_Equiavalent2', 'Delta', 'electronegativity', 'Bo', 'Cold Rolling Rate']]
    X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, stratify = df['stratify'], 
                                                    test_size=0.2, random_state=2361)

    x = np.linspace(250, 900)
    y = x
    
    model.fit(X_train, y_train)
    
    predictions_train = model.predict(X_train)
    predictions_test = model.predict(X_test)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    line1 = ax.plot(x, y, c='black')
    scatter2 = ax.scatter(y_train, predictions_train, c='blue', alpha=0.5, edgecolor='blue', linewidths=0.5, label='Measured UTS')
    scatter1 = ax.scatter(y_test, predictions_test, c='red',marker="D",alpha=0.5, edgecolor='red', linewidths=0.5, label='Predicted UTS')


    ax.set_xlim(250, 900)
    ax.set_ylim(250, 900)
    ax.legend()
    # ax.set_title('Predicted UTS (by RF) vs. Measured UTS')
    ax.set_xlabel('Measured UTS / MPa')
    ax.set_ylabel('Predicted UTS / MPa')
    
    if save_fig == True:
        plt.savefig(file_name, bbox_inches="tight", pad_inches=0.05)
        
    fig.show()
    
def plot_elongation(model, df, save_fig=False, file_name="sample.png"):

    df_y = df[['UTS']]
    df_X = df[['Ti', 'Mo_Equiavalent2', 'Delta', 'electronegativity', 'Bo', 'Cold Rolling Rate']]
    X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, stratify = df['stratify'], 
                                                    test_size=0.2, random_state=2361)

    x = np.linspace(0, 50)
    y = x
    
    model.fit(X_train, y_train)
    
    predictions_train = model.predict(X_train)
    predictions_test = model.predict(X_test)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    line1 = ax.plot(x, y, c='black')
    scatter2 = ax.scatter(y_train, predictions_train, c='blue', alpha=0.5, edgecolor='blue', linewidths=0.5, label='Measured Elongation')
    scatter1 = ax.scatter(y_test, predictions_test, c='red',marker="D",alpha=0.5, edgecolor='red', linewidths=0.5, label='Predicted Elongation')


    ax.set_xlim(0, 50)
    ax.set_ylim(0, 50)
    ax.legend()
    # ax.set_title('Predicted UTS (by RF) vs. Measured UTS')
    ax.set_xlabel('Measured Elongation / %')
    ax.set_ylabel('Predicted Elongation / %')
    
    if save_fig == True:
        plt.savefig(file_name, bbox_inches="tight", pad_inches=0.05)
        
    fig.show()