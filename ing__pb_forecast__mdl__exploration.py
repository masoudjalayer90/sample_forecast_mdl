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
from datetime import datetime
import sqlalchemy as db
import pyodbc


import requests
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
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

    '''
    conn = pyodbc.connect('Driver={SQL Server};'
                          'Server=PXLSTRRADAR,1771;'
                          'Database=RADAR_DM;'
                          'Trusted_Connection=yes;')

    cursor = conn.cursor()

    # model code look up
    cursor.execute("""
                      SELECT    *
                      INTO	    #model
                      FROM	    dm.dmr_model model
                      WHERE	    model.MODEL_DESC = 'CLIENT PRIMARY BANK'
                      """)

    #NTB
    cursor.execute("""
                        SELECT	a.client_key, eomonth(min(cal.CALENDAR_DATE)) as got_date 
                        INTO	#client_got
                        FROM	dm.dmr_event_client a
                                inner join dm.dmr_event b
                                on a.event_key = b.event_key and EVENT_NAME = 'GOT'
                                inner join dm.dmr_calendar cal 
                                on a.EVENT_CALENDAR_KEY = cal.CALENDAR_KEY
                        GROUP BY	a.client_key 
                        """)

    # hpb_1111
    cursor.execute("""
                        SELECT  metrics.CLIENT_KEY, convert(date,convert(varchar(10),metrics.CALENDAR_KEY ),112) as CALENDAR_KEY, MODEL_CODE, 
                                case 
                                    when datediff(mm,#client_got.got_date,convert(date,convert(varchar(10),metrics.CALENDAR_KEY ),112)) <= 2
                                    then 1 
                                    else 0 
                                end as NTB
                        INTO	    #hpb_1111_clients
                        FROM	    dm.dmr_model_metrics metrics
                                inner join #model
                        on metrics.MODEL_KEY = #model.MODEL_KEY
                                left join #client_got 
                        on metrics.client_key = #client_got.client_key
                        WHERE	    #model.MODEL_CODE = 'hpb_1111'
                      """)

    # <> hpb_1111
    cursor.execute("""
                        SELECT   metrics.CLIENT_KEY, convert(date,convert(varchar(10),metrics.CALENDAR_KEY ),112) as CALENDAR_KEY, MODEL_CODE,
                                case 
                                    when datediff(mm,#client_got.got_date,convert(date,convert(varchar(10),metrics.CALENDAR_KEY ),112)) <= 2
                                    then 1 
                                    else 0 
                                end as NTB
                        INTO      #non_pb_clients
                        FROM	    dm.dmr_model_metrics metrics
                                INNER JOIN #model
                        on metrics.MODEL_KEY = #model.MODEL_KEY
                                left join #client_got 
                        on metrics.client_key = #client_got.client_key
                        WHERE	    #model.MODEL_CODE <> 'hpb_1111'
                      """)

    # pb_gain table
    cursor.execute("""
                        SELECT    a.CALENDAR_KEY,a.MODEL_CODE, sum(customers) as customers,sum(NTB) as NTB
                        INTO	    #pb_gain
                        FROM	    (
                                SELECT  a.CALENDAR_KEY,b.model_code,count((a.CLIENT_KEY)) as customers,sum(a.NTB) as NTB
                                FROM	#hpb_1111_clients a
                                        INNER JOIN dm.dmr_calendar cal
                                        on a.calendar_key= convert(date,cal.calendar_date)
                                        INNER JOIN #non_pb_clients b
                                        on a.client_key = b.client_key
                                        and b.calendar_key = convert(date,convert(varchar(10),cal.prev_month_end_key),112)
                                GROUP BY	a.CALENDAR_KEY,b.model_code
                                UNION ALL
                                SELECT  a.CALENDAR_KEY,'Unknown' as model_code,count((a.CLIENT_KEY)) as customers,sum(a.NTB) as NTB
                                FROM	#hpb_1111_clients a
                                        INNER JOIN dm.dmr_calendar cal
                                        on a.calendar_key= convert(date,cal.calendar_date )
                                WHERE	NOT EXISTS
                                        (
                                        SELECT  'x'
                                        FROM	(
                                                SELECT	*
                                                FROM	#non_pb_clients
                                                UNION
                                                SELECT	*
                                                FROM	#hpb_1111_clients
                                                )t
                                        WHERE	t.client_key = a.client_key
                                        and t.calendar_key = convert(date,convert(varchar(10),cal.prev_month_end_key),112)
                                        )
                                GROUP BY a.CALENDAR_KEY
                                ) a
                        GROUP BY	a.CALENDAR_KEY,a.MODEL_CODE
                      """)


    # pb_loss table
    cursor.execute("""
                        SELECT	*
                        INTO	#pb_loss
                        FROM	(
                                SELECT	convert(date,convert(varchar(10),cal.NEXT_MONTH_END_KEY),112) as CALENDAR_KEY,ISNULL(a.MODEL_CODE,'Unknown') as MODEL_CODE, count(distinct(b.CLIENT_KEY)) as customers
                                FROM	#hpb_1111_clients b
                                        INNER JOIN dm.dmr_calendar cal
                                        on b.calendar_key = convert(date,cal.calendar_date)
                                        LEFT OUTER JOIN #non_pb_clients a
                                        ON a.calendar_key = convert(date,convert(varchar(10),cal.NEXT_MONTH_END_KEY),112)
                                        AND a.CLIENT_KEY=b.CLIENT_KEY
                                WHERE	not exists
                                        (
                                        SELECT	TOP(1) 'x'
                                        FROM	#hpb_1111_clients n
                                        WHERE	n.CLIENT_KEY=b.CLIENT_KEY
                                                AND n.CALENDAR_KEY = convert(date,convert(varchar(10),cal.NEXT_MONTH_END_KEY),112)
                                        )
                                GROUP BY	convert(date,convert(varchar(10),cal.NEXT_MONTH_END_KEY),112),ISNULL(a.MODEL_CODE,'Unknown')
                                ) a
                        WHERE	a.CALENDAR_KEY<=getdate()
                    """)

    # pb_gain
    cursor.execute("""
                        SELECT    DATEFROMPARTS(year(CALENDAR_KEY ),month(CALENDAR_KEY ), 01) as Dates, 
                                  case
                                    when MODEL_CODE = 'HPB_1101' then 'Xbuy_Gain'
                                    when MODEL_CODE = 'HPB_1110' then 'Salary_Gain'
                                    else 'Other_Gain'
                                  end as Model_Code, sum(customers) as Total_Gain, 
                                  sum(NTB) as NTB
                        FROM	    #pb_gain
                        WHERE     year(CALENDAR_KEY) > 2014
                        GROUP BY  DATEFROMPARTS(year(CALENDAR_KEY ),month(CALENDAR_KEY ), 01), case
                                    when MODEL_CODE = 'HPB_1101' then 'Xbuy_Gain'
                                    when MODEL_CODE = 'HPB_1110' then 'Salary_Gain'
                                    else 'Other_Gain'
                                  end 
                      """)


    gain_labels = ['Dates','Model_Code' , 'Total_Gain', 'Total_NTB']
    gain_q = cursor.fetchall()
    gain_list = []

    # pb_loss
    cursor.execute("""
                        SELECT    DATEFROMPARTS(year(CALENDAR_KEY ),month(CALENDAR_KEY ), 01) as Dates, 
                                  case
                                    when MODEL_CODE = 'HPB_1101' then 'Xbuy_Loss'
                                    when MODEL_CODE = 'HPB_1110' then 'Salary_Loss'
                                    else 'Other_Loss'
                                  end as Model_Code,
                                   sum(customers) as Total_Loss
                        FROM	    #pb_loss
                        GROUP BY  DATEFROMPARTS(year(CALENDAR_KEY ),month(CALENDAR_KEY ), 01), 
                                    case
                                        when MODEL_CODE = 'HPB_1101' then 'Xbuy_Loss'
                                        when MODEL_CODE = 'HPB_1110' then 'Salary_Loss'
                                        else 'Other_Loss'
                                    end
                        """)

    loss_labels = ['Dates','Model_Code', 'Total_Loss']
    loss_q = cursor.fetchall()
    loss_list = []

    # oe_actuals
    cursor.execute("""
                    SELECT	DATEFROMPARTS(year(OPENED_DATE),month(OPENED_DATE), 01) as Dates, count(*) as OE_opens
                    FROM	[RADAR].[dm].[DMR_ACCOUNT_SAVINGS_CURRENT]
                    WHERE	RADAR_PRODUCT_GROUP_LABEL = 'OE'
                            and year(opened_Date) > 2014
                            and EOMONTH(opened_Date) <= EOMONTH(dateadd(month,-1,GETDATE()))
                    GROUP BY	DATEFROMPARTS(year(OPENED_DATE),month(OPENED_DATE), 01)
                    order by	1
                     """)

    oe_labels = ['Dates', 'OE_opens']
    oe_q = cursor.fetchall()
    oe_list = []

    def list_to_df(query, lists, labels):
        """ this function convert a lists of pyocdb components into a pandas DF"""
        for row in query:
            lists.append(list(row))
        return pd.DataFrame(lists, columns=labels)

    pb_gain = list_to_df(gain_q, gain_list, gain_labels)
    pb_loss = list_to_df(loss_q, loss_list, loss_labels)
    oe_open = list_to_df(oe_q, oe_list, oe_labels)


    '''

    #############################

    # pb_gain = pd.read_excel("C:/Users/mjalayer/PycharmProjects/pb_forecast/data_in/test_gain.xlsx")
    pb_loss = pd.read_excel("C:/Users/mjalayer/PycharmProjects/pb_forecast/data_in/test_loss.xlsx")
    oe_open = pd.read_excel("C:/Users/mjalayer/PycharmProjects/pb_forecast/data_in/OE_Opens_exog.xlsx")
    oe_forecast = pd.read_excel("C:/Users/mjalayer/PycharmProjects/pb_forecast/data_in/OE_Opens_exog_forecast.xlsx")
    pb_gain_split = pd.read_excel("C:/Users/mjalayer/PycharmProjects/pb_forecast/data_in/PB_gain_split.xlsx")

    pb_gain = pb_gain_split.pivot(index='Dates',columns='Model_Code')['Total_Gain']
    pb_gain['Total_Gain'] = pb_gain.sum(axis=1)

    pb_ntb = pb_gain_split.pivot(index='Dates', columns='Model_Code')['New to Bank']
    pb_ntb['Total_NTB'] = pb_ntb.sum(axis=1)
    nbt_columns = ['Other_Gain_NTB', 'Salary_Gain_NTB', 'Xbuy_Gain_NTB', 'Total_NTB']
    pb_ntb.columns = nbt_columns

    pb_loss = pb_loss.pivot(index='Dates', columns='Model_Code')['Total_Loss']
    pb_loss['Total_Loss'] = pb_loss.sum(axis=1)

    pb_gain = pb_gain.sort_values('Dates')
    pb_loss = pb_loss.sort_values('Dates')
    oe_open = oe_open.sort_values('Dates')
    oe_forecast = oe_forecast.sort_values('Dates')

    oe_open = oe_open.set_index('Dates')
    oe_fc1 = oe_forecast.set_index('Dates')

    g = pb_gain['Total_Gain'].resample('MS').mean()
    l = pb_loss['Total_Loss'].resample('MS').mean()
    oe = oe_open['OE_opens'].resample('MS').mean()

    # plot pb data
    g.plot(figsize=(15, 10))
    l.plot(figsize=(15, 10))
    oe.plot(figsize=(15, 10))
    plt.title("PB Actuals Gain and Loss, OE")
    plt.legend()
    # plt.show()

    # from statsmodels.tsa.stattools import adfuller
    # # gain adf
    # result = adfuller(g['2017-01-01':])
    # print('Gain: ADF Statistic: %f' % result[0])
    # print('p-value: %f' % result[1])
    # print('Critical Values:')
    # for key, value in result[4].items():
    #     print('\t%s: %.3f' % (key, value))
    #
    # result = adfuller(g['2017-01-01':].diff().dropna())
    # print('Gain Diff: ADF Statistic: %f' % result[0])
    # print('p-value: %f' % result[1])
    # print('Critical Values:')
    # for key, value in result[4].items():
    #     print('\t%s: %.3f' % (key, value))
    #
    # # loss adf
    # result = adfuller(l)
    # print('Loss: ADF Statistic: %f' % result[0])
    # print('p-value: %f' % result[1])
    # print('Critical Values:')
    # for key, value in result[4].items():
    #     print('\t%s: %.3f' % (key, value))
    #
    # result = adfuller(l.diff().diff().dropna())
    # print('Loss Diff: ADF Statistic: %f' % result[0])
    # print('p-value: %f' % result[1])
    # print('Critical Values:')
    # for key, value in result[4].items():
    #     print('\t%s: %.3f' % (key, value))
    #
    # # oe adf
    # result = adfuller(oe['2017-01-01':])
    # print('oe: ADF Statistic: %f' % result[0])
    # print('p-value: %f' % result[1])
    # print('Critical Values:')
    # for key, value in result[4].items():
    #     print('\t%s: %.3f' % (key, value))
    #
    # result = adfuller(oe['2017-01-01':].diff().dropna())
    # print('oe Diff: ADF Statistic: %f' % result[0])
    # print('p-value: %f' % result[1])
    # print('Critical Values:')
    # for key, value in result[4].items():
    #     print('\t%s: %.3f' % (key, value))



    # # three components of the signal
    # from pylab import rcParams
    # rcParams['figure.figsize'] = 15,10
    # decomposition_gain = sm.tsa.seasonal_decompose(g, model='additive')
    # fig_g = decomposition_gain.plot()
    # plt.title("Decomposition PB Gain")
    # # plt.show(fig_g)
    #
    decomposition_loss = sm.tsa.seasonal_decompose(l, model='additive')
    fig_l = decomposition_loss.plot()
    plt.title("Decomposition PB Loss")
    # plt.show(fig_l)

    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

    grid_list_g = []
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            grid_results = {}
            grid_results['param'] = param
            grid_results['param_seasonal'] = param_seasonal
            try:
                mod = sm.tsa.statespace.SARIMAX(g['2017-01-01':],
                                                exog=oe['2017-01-01':],
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

    mod_g = sm.tsa.statespace.SARIMAX(g['2017-01-01':],
                                      order=param_min_g['param'],
                                      seasonal_order=param_min_g['param_seasonal'],
                                      # order=(1, 1, 1),
                                      # seasonal_order=(1, 1, 0, 12),
                                      exog=oe['2017-01-01':],
                                      enforce_stationarity=False,
                                      enforce_invertibility=False)

    results_g = mod_g.fit()
    print(results_g.summary().tables[1])
    # results_g.plot_diagnostics(figsize=(15,10))
    # plt.show()

    mod_l = sm.tsa.statespace.SARIMAX(l,
                                      order=param_min_l['param'],
                                      seasonal_order=param_min_l['param_seasonal'],
                                      # order=(1, 1, 1),
                                      # seasonal_order=(1, 1, 1, 12),
                                      enforce_stationarity=False,
                                      enforce_invertibility=False)

    results_l = mod_l.fit()
    print(results_l.summary().tables[1])
    # results_l.plot_diagnostics(figsize=(15,10))
    # plt.show()

    pred_g = results_g.get_prediction(start=pd.to_datetime('2019-01-01'),
                                      exog=oe['2017-01-01':],
                                      dynamic=False)

    pred_ci_g = pred_g.conf_int()

    ax = g['2017-01-01':].plot(label='Observed')
    pred_g.predicted_mean.plot(ax=ax, label='One-step ahead Forecast',
                              alpha=.7, figsize=(15,10))

    ax.fill_between(pred_ci_g.index,
                    pred_ci_g.iloc[:,0],
                    pred_ci_g.iloc[:,1], color='k', alpha=.2)

    plt.xlabel('Date')
    plt.ylabel('PB_Gain')
    plt.legend()
    plt.title("PB Gain Forecast Validation")
    # plt.show()

    pred_l = results_l.get_prediction(start=pd.to_datetime('2019-01-01'),
                                      dynamic=False)
    pred_ci_l = pred_l.conf_int()

    ax = l.plot(label='Observed')
    pred_l.predicted_mean.plot(ax=ax, label='One-step ahead Forecast',
                               alpha=.7, figsize=(15,10))

    ax.fill_between(pred_ci_l.index,
                    pred_ci_l.iloc[:, 0],
                    pred_ci_l.iloc[:, 1], color='k', alpha=.2)
    plt.xlabel('Date')
    plt.ylabel('PB_Loss')
    plt.legend()
    plt.title("PB Loss Forecast Validation")
    # plt.show()

    g_forecasted = pred_g.predicted_mean
    g_truth = g['2017-01-01':]

    l_forecasted = pred_l.predicted_mean
    l_truth = l['2017-01-01':]

    # MSE
    mse_g = ((g_forecasted - g_truth)**2).mean()
    mse_l = ((l_forecasted - l_truth) ** 2).mean()

    print('The MSE of the forecast_gain is {}'.format(round(mse_g, 2)))
    print('The MSE of the forecast_loss is {}'.format(round(mse_l, 2)))

    # RMSE
    print('The Root Mean Squared Error of our forecasts_gain is {}'.format(round(np.sqrt(mse_g), 2)))
    print('The Root Mean Squared Error of our forecasts_loss is {}'.format(round(np.sqrt(mse_l), 2)))

    today = datetime.today()
    this_month = datetime(today.year, today.month, 1)

    year_end = datetime(today.year, 12, 1)

    pred_g_exog = results_g.get_prediction(start=this_month,
                                           end=year_end,
                                           exog=oe_fc1,
                                           dynamic=False)

    pred_ci_g = pred_g_exog.conf_int(alpha=0.1)

    pred_l = results_l.get_prediction(start=this_month,
                                            end=year_end,
                                            dynamic=False)
    pred_ci_l = pred_l.conf_int(alpha=0.1)

    pred_g_output = pd.DataFrame(pred_g_exog.predicted_mean)
    pred_g_output = pred_g_output.reset_index(drop=False)
    g_columns = ['Dates', 'Total_Gain']
    pred_g_output.columns = g_columns
    # pred_g_output.to_csv('C:/Users/mjalayer/PycharmProjects/pb_forecast/data_out/pred_g_output.csv')

    pred_l_output = pd.DataFrame(pred_l.predicted_mean)
    pred_l_output = pred_l_output.reset_index(drop=False)
    l_columns = ['Dates', 'Total_Loss']
    pred_l_output.columns = l_columns
    # pred_l_output.to_csv('C:/Users/mjalayer/PycharmProjects/pb_forecast/data_out/pred_l_output.csv')

    pb_gain['Forecast_Actuals'] = 'Actuals'
    pred_g_output['Forecast_Actuals'] = 'Forecast'

    pb_gain = pb_gain.reset_index(drop=False)
    gain_frame = [pb_gain, pred_g_output]
    df_acq = pd.concat(gain_frame, axis=0, ignore_index=True)

    pb_loss = pb_loss.reset_index(drop=False)
    loss_frame = [pb_loss, pred_l_output]
    df_att = pd.concat(loss_frame, axis=0, ignore_index=True)

    oe_open = oe_open.reset_index(drop=False)
    open_frame = [oe_open, oe_forecast]
    df_oe = pd.concat(open_frame, axis=0, ignore_index=True)

    pb_ntb = pb_ntb.reset_index(drop=False)

    total = pd.merge(df_acq, df_att, on='Dates')
    total = pd.merge(total, pb_ntb, on='Dates', how='left')
    total = pd.merge(total, df_oe, on='Dates')
    total['Net_PB'] = total['Total_Gain'] - total['Total_Loss']

    total_columns = ['Dates', 'Forecast_Actuals', 'Other_Gain', 'Salary_Gain',
                     'Acquisition Actuals_Forecasts',	'Xbuy_Gain', 'Other_Loss',
                     'Salary_Loss', 'Attrition Actuals_Forecast', 'Xbuy_Loss',
                     'Other_Gain_NTB', 'Salary_Gain_NTB', 'Xbuy_Gain_NTB', 'Total_NTB',
                     'OE_opens', 'Net_PB']
    total.columns = total_columns

    total.to_csv('C:/Users/mjalayer/PycharmProjects/pb_forecast/data_out/output_march.csv', index=False)

    ax = g.plot(label='observed', figsize=(15, 10))
    pred_g_exog.predicted_mean.plot(ax=ax, label='Forecast_Gain')
    ax.fill_between(pred_ci_g.index,
                    pred_ci_g.iloc[:, 0],
                    pred_ci_g.iloc[:, 1], color='k', alpha=.25)
    plt.xlabel('Date')
    plt.ylabel('PB_Gain')
    plt.legend()
    plt.title("PB Gain Projected Forecast")
    # plt.show()

    ax = l.plot(label='observed', figsize=(15, 10))
    pred_l.predicted_mean.plot(ax=ax, label='Forecast_Loss')
    ax.fill_between(pred_ci_l.index,
                    pred_ci_l.iloc[:, 0],
                    pred_ci_l.iloc[:, 1], color='k', alpha=.25)
    plt.xlabel('Date')
    plt.ylabel('PB_Loss')
    plt.legend()
    plt.title("PB Loss Projected Forecast")
    # plt.show()


if __name__ == "__main__":
    main()
