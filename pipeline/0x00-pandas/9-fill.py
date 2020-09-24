#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df = df.drop(columns=["Weighted_Price"])
col = ["High", "Low", "Open"]
df["Close"].fillna(method="backfill", inplace=True)
df[col]
#df[col] = df[col].loc[df[col].isna]
#df[col] = df[col].loc[df[col].isna]
print(df.head())
print(df.tail())
