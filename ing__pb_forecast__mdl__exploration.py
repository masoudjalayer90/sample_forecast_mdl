import configparser
import logging, logging.config
import os
from pathlib import Path
from pprint import pprint
import sys
import time

import itertools
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
import warnings

warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')

matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'


# import globals
# from globals import timer, main_wrapper

__author__ = "Masoud Jalayer"
__created__ = "2019-12-09"
__copyright__ = "Copyright 2019, CHE Proximity"
__credits__ = ["Masoud Jalayer"]
__status__ = "Development" # "Development", "Prototype", or "Production"
__version__ = "0.1.0"
__maintainer__ = "Masoud Jalayer"
__email__ = "masoud.jalayer@cheproximity.com.au"


def main():

    # hist_pb = pd.read_excel("C:/Users/mjalayer/PycharmProjects/pb_forecast/data/historical_PB_figures.xlsx")
    pb_gain = pd.read_excel("C:/Users/mjalayer/PycharmProjects/pb_forecast/data_in/PB_gain_totals.xlsx")
    pb_loss = pd.read_excel("C:/Users/mjalayer/PycharmProjects/pb_forecast/data_in/PB_loss_totals.xlsx")

    # print(hist_pb)

    # min and max dates
    # print(hist_pb['Dates'].min(), hist_pb['Dates'].max())

    # hist_pb = hist_pb.sort_values('Dates')
    pb_gain = pb_gain.sort_values('Dates')
    pb_loss = pb_loss.sort_values('Dates')

    # null values in the data
    # print(hist_pb.isnull().sum())

    # grouping sales by dates
    # hist_pb = hist_pb.groupby('Dates')['PB_Gain'].sum().reset_index()
    pb_gain = pb_gain.groupby('Dates')['Total_Gain'].sum().reset_index()
    pb_loss = pb_loss.groupby('Dates')['Total_Loss'].sum().reset_index()

    # set Date as index
    # hist_pb = hist_pb.set_index('Dates')
    pb_gain = pb_gain.set_index('Dates')
    pb_loss = pb_loss.set_index('Dates')


    # y = hist_pb['PB_Gain'].resample('MS').mean()
    g = pb_gain['Total_Gain'].resample('MS').mean()
    l = pb_loss['Total_Loss'].resample('MS').mean()

    # plot pb data
    g.plot(figsize=(15,6))
    l.plot(figsize=(15,6))
    plt.title("PB Actuals Gain and Loss")
    plt.legend()
    plt.show()

    # three components of the sale data
    from pylab import rcParams
    rcParams['figure.figsize'] = 18, 8
    decomposition_gain = sm.tsa.seasonal_decompose(g, model='additive')
    fig_g = decomposition_gain.plot()
    plt.title("Decomposition PB Gain")
    plt.show(fig_g)

    decomposition_loss = sm.tsa.seasonal_decompose(l, model='additive')
    fig_l = decomposition_loss.plot()
    plt.title("Decomposition PB Loss")
    plt.show(fig_l)

    p = d = q = range(0,2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

    grid_list_g = []
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            grid_results = {}
            grid_results['param'] = param
            grid_results['param_seasonal'] = param_seasonal
            try:
                mod = sm.tsa.statespace.SARIMAX(g,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)
                results = mod.fit(disp=False)
                grid_results['aic'] = results.aic
            except:
                continue
            grid_list_g.append(grid_results)

    df_grid_g = pd.DataFrame(grid_list_g)
    df_grid_g = df_grid_g.sort_values('aic').reset_index()
    param_min_g = df_grid_g.iloc[0]
    print(param_min_g)

    grid_list_l = []
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            grid_results = {}
            grid_results['param'] = param
            grid_results['param_seasonal'] = param_seasonal
            try:
                mod = sm.tsa.statespace.SARIMAX(l,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)
                results = mod.fit(disp=False)
                grid_results['aic'] = results.aic
            except:
                continue
            grid_list_l.append(grid_results)

    df_grid_l = pd.DataFrame(grid_list_l)
    df_grid_l = df_grid_l.sort_values('aic').reset_index()
    param_min_l = df_grid_l.iloc[0]
    print(param_min_l)

    mod_g = sm.tsa.statespace.SARIMAX(g,
                                    order=param_min_g["param"],
                                    seasonal_order=param_min_g["param_seasonal"],
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)

    results_g = mod_g.fit()
    print(results_g.summary().tables[1])
    # results_g.plot_diagnostics(figsize=(16,8))
    plt.show()

    mod_l = sm.tsa.statespace.SARIMAX(l,
                                    order=param_min_l["param"],
                                    seasonal_order=param_min_l["param_seasonal"],
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)

    results_l = mod_l.fit()
    print(results_l.summary().tables[1])
    # results_l.plot_diagnostics(figsize=(16,8))
    plt.show()

    pred_g = results_g.get_prediction(start=pd.to_datetime('2019-01-01'),
                                  dynamic=False)
    pred_ci_g = pred_g.conf_int()

    ax = g.plot(label='Observed')
    pred_g.predicted_mean.plot(ax=ax, label='One-step ahead Forecast',
                              alpha=.7, figsize=(14,7))

    ax.fill_between(pred_ci_g.index,
                    pred_ci_g.iloc[:,0],
                    pred_ci_g.iloc[:,1], color='k', alpha=.2)
    plt.xlabel('Date')
    plt.ylabel('PB_Gain')
    plt.legend()
    plt.title("PB Gain Forecast Validation")
    plt.show()

    pred_l = results_l.get_prediction(start=pd.to_datetime('2019-01-01'),
                                  dynamic=False)
    pred_ci_l = pred_l.conf_int()

    ax = l.plot(label='Observed')
    pred_l.predicted_mean.plot(ax=ax, label='One-step ahead Forecast',
                              alpha=.7, figsize=(14,7))

    ax.fill_between(pred_ci_l.index,
                    pred_ci_l.iloc[:,0],
                    pred_ci_l.iloc[:,1], color='k', alpha=.2)
    plt.xlabel('Date')
    plt.ylabel('PB_Loss')
    plt.legend()
    plt.title("PB Loss Forecast Validation")
    plt.show()

    g_forecasted = pred_g.predicted_mean
    g_truth = g['2017':]

    l_forecasted = pred_l.predicted_mean
    l_truth = l['2017':]

    # MSE
    mse_g = ((g_forecasted - g_truth)**2).mean()
    mse_l = ((l_forecasted - l_truth) ** 2).mean()

    print('The MSE of the forecast_gain is {}'.format(round(mse_g,2)))
    print('The MSE of the forecast_loss is {}'.format(round(mse_l,2)))

    # RMSE
    print('The Root Mean Squared Error of our forecasts_gain is {}'.format(round(np.sqrt(mse_g), 2)))
    print('The Root Mean Squared Error of our forecasts_loss is {}'.format(round(np.sqrt(mse_l), 2)))

    pred_uc_g = results_g.get_forecast(steps=36)
    pred_ci_g = pred_uc_g.conf_int()

    pred_g_output = pd.DataFrame(pred_uc_g.predicted_mean)
    pred_g_output.to_csv('C:/Users/mjalayer/PycharmProjects/pb_forecast/data_out/pred_g_output.csv')

    pred_uc_l = results_l.get_forecast(steps=36)
    pred_ci_l = pred_uc_l.conf_int()

    pred_l_output = pd.DataFrame(pred_uc_l.predicted_mean)
    pred_l_output.to_csv('C:/Users/mjalayer/PycharmProjects/pb_forecast/data_out/pred_l_output.csv')


    ax = g.plot(label='observed', figsize=(14,7))
    pred_uc_g.predicted_mean.plot(ax=ax, label='Forecast_Gain')
    ax.fill_between(pred_ci_g.index,
                    pred_ci_g.iloc[:,0],
                    pred_ci_g.iloc[:,1], color='k', alpha=.25)
    plt.xlabel('Date')
    plt.ylabel('PB_Gain')
    plt.legend()
    plt.title("PB Gain Projected Forecast")
    plt.show()

    ax = l.plot(label='observed', figsize=(14,7))
    pred_uc_l.predicted_mean.plot(ax=ax, label='Forecast_Loss')
    ax.fill_between(pred_ci_l.index,
                    pred_ci_l.iloc[:,0],
                    pred_ci_l.iloc[:,1], color='k', alpha=.25)
    plt.xlabel('Date')
    plt.ylabel('PB_Loss')
    plt.legend()
    plt.title("PB Loss Projected Forecast")
    plt.show()

if __name__ == "__main__":
    main()
