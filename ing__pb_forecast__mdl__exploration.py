#!/usr/bin/env python3

"""

Docstring long description
"""
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
import yaml
import warnings

warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')

matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'


import globals
from globals import timer, main_wrapper

__author__ = "Masoud Jalayer"
__created__ = "2019-12-09"
__copyright__ = "Copyright 2019, CHE Proximity"
__credits__ = ["Masoud Jalayer"]

__version__ = "0.1.0"
__maintainer__ = "Masoud Jalayer"
__email__ = "masoud.jalayer@cheproximity.com.au"
__status__ = "Development" # "Development", "Prototype", or "Production"


# @main_wrapper
def main():
    df_in = pd.read_excel(globals.DATA_RAW_PATH/'Sample - Superstore.xls')
    furniture = df_in.loc[df_in['Category'] == 'Furniture']

    # min and max dates
    # print(furniture['Order Date'].min(), furniture['Order Date'].max())

    # drop columns not needed
    cols = ['Row ID', 'Order ID', 'Ship Date', 'Ship Mode', 'Customer ID', 'Customer Name', 'Segment', 'Country', 'City', 'State', 'Postal Code', 'Region', 'Product ID', 'Category', 'Sub-Category', 'Product Name', 'Quantity', 'Discount', 'Profit']
    furniture.drop(cols, axis=1, inplace=True)
    furniture = furniture.sort_values('Order Date')

    # null values in the data
    # print(furniture.isnull().sum())

    # grouping sales by dates
    furniture = furniture.groupby('Order Date')['Sales'].sum().reset_index()

    # set Date as index
    furniture = furniture.set_index('Order Date')

    y = furniture['Sales'].resample('MS').mean()

    #plot sale data
    # y.plot(figsize=(15,6))
    # plt.show()

    # three components of the sale data
    from pylab import rcParams
    rcParams['figure.figsize'] = 18, 8
    decomposition = sm.tsa.seasonal_decompose(y, model='additive')
    fig = decomposition.plot()
    plt.show(fig)

    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

    grid_list = []
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            grid_results = {}
            grid_results['param'] = param
            grid_results['param_seasonal'] = param_seasonal
            try:
                mod = sm.tsa.statespace.SARIMAX(y,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)
                results = mod.fit(disp=False)
                grid_results['aic'] = results.aic
            except:
                continue
            grid_list.append(grid_results)

    df_grid = pd.DataFrame(grid_list)
    df_grid = df_grid.sort_values('aic').reset_index()
    param_min = df_grid.iloc[0]
    print(param_min)

    mod = sm.tsa.statespace.SARIMAX(y,
                                    order=param_min['param'],
                                    seasonal_order=param_min['param_seasonal'],
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)


    results = mod.fit()
    print(results.summary().tables[1])
    results.plot_diagnostics(figsize=(16,8))
    plt.show()


    pred = results.get_prediction(start=pd.to_datetime('2017-01-01'),
                                  dynamic=False)
    pred_ci = pred.conf_int()

    ax = y.plot(label='Observed')
    pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast',
                              alpha=.7, figsize=(14,7))

    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:,0],
                    pred_ci.iloc[:,1], color='k', alpha=.2)
    plt.xlabel('Date')
    plt.ylabel('Furniture Sale')
    plt.legend()
    plt.show()

    y_forecasted = pred.predicted_mean
    y_truth = y['2017':]

    # MSE
    mse = ((y_forecasted - y_truth)**2).mean()
    print('The MSE of the forecast is {}'.format(round(mse,2)))
    # RMSE
    print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))

    pred_uc = results.get_forecast(steps=100)
    pred_ci = pred_uc.conf_int()

    ax = y.plot(label='observed', figsize=(14,7))
    pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:,0],
                    pred_ci.iloc[:,1], color='k', alpha=.25)
    plt.xlabel('Date')
    plt.ylabel('Furniture Sale')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()