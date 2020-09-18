#!/usr/bin/env python3
""" Time Series Forecasting - preprocess"""
import numpy as np
# import pandas as pd


def preprocessor(datafile, days):
    """ pre-process dataset:
        fills empty values, re-sample the time series
        and split the data
        - datafile: string dataset location
        - days: days to take from the dataset (the last)
        returns:
            df_W is the data for univarite
            df_mul is the data for multivariate
    """
    df = pd.read_csv(datafile)
    df = df.iloc[-days * 24 * 60:]
    df = df.drop(['Volume_(Currency)'], axis=1)  # volume dollar deleted
    df['Datetime'] = pd.to_datetime(df['Timestamp'], unit='s')
    df = df.drop(['Timestamp'], axis=1)
    df = df.set_index('Datetime')
    df = df.interpolate()  # completes null spots
    df.columns = df.columns.str.replace('(', '')
    df.columns = df.columns.str.replace(')', '')
    df_mul = pd.DataFrame()  # multivariate
    df_mul['High'] = df.High.resample('H').max()
    df_mul['Low'] = df.Low.resample('H').min()
    df_mul['Weighted_Price'] = df.Weighted_Price.resample('H').mean()
    df_mul['Volume_BTC'] = df.Volume_BTC.resample('H').sum()
    df_W = df.Weighted_Price.resample('H').mean()  # univariate

    df_W.plot(subplots=True)

    return df_W, df_mul
