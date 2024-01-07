import numpy as np
import pandas as pd
import datatable as dt
import dask
import powerlaw

from library_data import *

def basic_stats(df, name_asset) :
    """Print basic statistics of a given dataframe.

    :param df: Dataframe with the historical data.
    :type df: DataFrame
    :param name_asset: Name of the asset.
    :type name_asset: String
    """
    print(f"----- {name_asset} daily - Close Price -----")
    print(f"Mean :\t {df['Close'].mean():.2f}")
    print(f"Std:\t{df['Close'].std() : .2f}")
    print(f"Ratio :\t {100*df['Close'].std()/df['Close'].mean():.2f}%")
    print()

def compare_btc_sp500_aapl(btc_daily) :
    """Compare daily price of Bitcoin, S&P500 and Apple.

    :param btc_daily: Daily price of Bitcoin.
    :type btc_daily: DataFrame
    """
    #Print basic statistics for BTC 
    basic_stats(btc_daily, "BTC")

    #Get daily price of S&P500
    gspc_daily = get_daily("^GSPC", btc_daily.index.min())
    gspc_daily = gspc_daily.loc[:btc_daily.index.max()]

    #Print basic statistics for S&P500
    basic_stats(gspc_daily, "BTC")

    #Get daily price of Apple
    aapl_daily = get_daily("AAPL", btc_daily.index.min())
    aapl_daily = aapl_daily.loc[:btc_daily.index.max()]

    #Print basic statistics for Apple
    basic_stats(aapl_daily, "BTC")

def powerlaw_estimation(df) :
    """Estimate powerlaw parameters for log-returns of a given asset.

    :param df: Dataframe with log-returns.
    :type df: DataFrame
    :return: Powerlaw fit.
    :rtype: Powerlaw fit object
    """
    #Group data by 30 minutes
    df = df.groupby(pd.Grouper(freq='30T')).mean()
    abslogrets = np.abs(df['Logrets'])

    #Fit powerlaw
    myfit = powerlaw.Fit(abslogrets)
    return myfit

def powerlaw_compare(fit, name) :
    """Compare powerlaw fit with exponential and lognormal distributions.

    :param fit: Powerlaw fit.
    :type fit: Powerlaw fit object
    :param name: Name of the asset.
    :type name: String
    :return: Dataframe containing results of statistical tests comparing powerlaw distribution with other distributions.
    :rtype: DataFrame
    """

    distributions = ['lognormal', 'exponential']
    res = dict()

    #Compare powerlaw fit with other distributions via statistical tests
    for distrib in distributions : 
        res_test = fit.distribution_compare("power_law", distrib, normalized_ratio=True)
        res[distrib] = [res_test[0], res_test[1], (res_test[0]>0 and res_test[1]<0.05)]
    
    res_df = pd.DataFrame(res, index=['z-score', 'p-value', 'powerlaw favored ?'])
    res_df['asset'] = name
    return res_df

def powerlaw_estimation_all(df_btc, df_eth, df_ada, df_ltc) :
    """Compare powerlaw fit with exponential and lognormal distributions of BTC, ADA, LTC and ETH.

    :param btc_df: Dataframe containing log-returns for BTC. 
    :type btc_df: DataFrame
    :param eth_df: Dataframe containing log-returns for ETH. 
    :type eth_df: DataFrame
    :param ada_df: Dataframe containing log-returns for ADA. 
    :type ada_df: DataFrame
    :param ltc_df: Dataframe containing log-returns for LTC. 
    :type ltc_df: DataFrame
    :return: Dataframe containing results of statistical tests comparing powerlaw distribution with other distributions.
    :rtype: DataFrame
    """
    cryptos = ['btc', 'ltc', 'ada', 'eth']

    fitted = dict()
    distrib_df = []

    #For each crypto, fit powerlaw and compare with other distributions
    for crypto in cryptos:
        #Fit parameters
        fit = powerlaw_estimation(locals()[f'df_{crypto}'])
        fitted[crypto] = [fit.alpha, fit.xmin]  
        #Compare distributions
        try : 
            distrib_df.append(powerlaw_compare(fit, crypto))
        except :
            print(f"Error with {crypto}")
            distrib_df.append(pd.DataFrame())

    fitted_df = pd.DataFrame(fitted, index=['alpha', 'xmin'])
    distrib_df = pd.concat(distrib_df, axis=0)
    distrib_df.reset_index(inplace=True)
    distrib_df.set_index(["asset", "index"], inplace=True)

    return fitted_df, distrib_df
        
