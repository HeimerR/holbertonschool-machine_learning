#!/usr/bin/env python3
""" Time Series Forecasting - preprocess"""
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False
f = './coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv'
df = pd.read_csv(f)
days = 730
df = df.iloc[-days*24*60:]
df = df.drop(['Volume_(Currency)'], axis=1)
df['Datetime'] = pd.to_datetime(df['Timestamp'], unit='s')
df = df.drop(['Timestamp'], axis=1)
df = df.set_index('Datetime')
df = df.interpolate()
df.columns = df.columns.str.replace('(', '')
df.columns = df.columns.str.replace(')', '')
df_mul = pd.DataFrame()
df_mul['High'] = df.High.resample('H').max()
df_mul['Low'] = df.Low.resample('H').min()
df_mul['Weighted_Price'] = df.Weighted_Price.resample('H').mean()
df_mul['Volume_BTC'] = df.Volume_BTC.resample('H').sum()
df_W = df.Weighted_Price.resample('H').mean()

df_W.plot(subplots=True)
df = df_W.values