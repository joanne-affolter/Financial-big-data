import numpy as np
import pandas as pd
import dask 


def logreturns(df, col):
    """
    Compute logreturns of a given column of a dataframe.
    Args:
        df (DataFrame): Dataframe with the historical data.
        col (str): Column name.
    Returns:
        df (DataFrame): Dataframe with the historical data and the logreturns.
    """
    df['Logrets'] = np.log(df[col]).diff()
    df.dropna(subset="Logrets", inplace=True)
    return df

@dask.delayed
def autocorrelation(df, col_name, lag) : 
    """
    Compute autocorrelation for a given lag.
    Args:
        df (DataFrame): Dataframe with the historical data.
        lag (int): Lag.
    Returns:
        autocorr (float): Autocorrelation.
    """
    autocorr = df[col_name].autocorr(lag=lag)
    return autocorr

def autocorrelation_dask(df, col_name, lags) :
    """
    Compute autocorrelation for a list of lags.
    Args:
        df (DataFrame): Dataframe with the historical data.
        lags (list): List of lags.
    Returns:
        alldata (list): List of autocorrelations at different lags.
    """
        
    allpromises=[autocorrelation(df, col_name, lag) for lag in lags]
    alldata=dask.compute(allpromises)[0]
    return alldata


def volatility(data, rolling_window_size):
    """
    Compute volatility of Logreturns of a dataframe.
    """
    df = data.copy()
    df['Volatility'] = df["Logrets"].rolling(rolling_window_size).std()
    df.dropna(subset="Volatility", inplace=True)
    return df[["Volatility"]]