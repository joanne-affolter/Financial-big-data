import numpy as np
import pandas as pd
import dask 


def logreturns(df, col):
    """Generate logreturns for a given column.

    :param df: Dataframe with the historical data.
    :type df: DataFrame
    :param col: Column name.
    :type col: String
    :return: Dataframe with the logreturns.
    :rtype: DataFrame
    """
    df['Logrets'] = np.log(df[col]).diff()
    df.dropna(subset="Logrets", inplace=True)
    return df

@dask.delayed
def autocorrelation(df, col_name, lag) : 
    """Compute autocorrelation for a given lag.

    :param df: Dataframe with the historical data.
    :type df: DataFrame
    :param col_name: Column name.
    :type col_name: String
    :param lag: Lag.
    :type lag: Int
    :return: Autocorrelation.
    :rtype: List of floats
    """
    autocorr = df[col_name].autocorr(lag=lag)
    return autocorr

def autocorrelation_dask(df, col_name, lags) :
    """Compute autocorrelation for a list of lags.

    :param df: Dataframe with the historical data.
    :type df: DataFrame
    :param col_name: Column name.
    :type col_name: String
    :param lags: List of lags.
    :type lags: List of ints
    :return: Autocorrelation.
    :rtype: List of floats
    """
    allpromises=[autocorrelation(df, col_name, lag) for lag in lags]
    alldata=dask.compute(allpromises)[0]
    return alldata

def volatility(data, rolling_window_size):
    """Compute volatility for a given rolling window size.

    :param data: Dataframe with the log-returns data.
    :type data: DataFrame
    :param rolling_window_size: Rolling window size.
    :type rolling_window_size: Int
    :return: Volatility.
    :rtype: DataFrame
    """
    df = data.copy()
    df['Volatility'] = df["Logrets"].rolling(rolling_window_size).std()
    df.dropna(subset="Volatility", inplace=True)
    return df[["Volatility"]]