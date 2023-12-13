import numpy as np
import pandas as pd
import datatable as dt
import dask
import powerlaw

from library_data import *

def basic_stats(df, name_asset) :
    """
    Print basic statistics of the dataframe.
    Args:
        df (DataFrame): Dataframe with the historical data.
        name_asset (str): Name of the asset.
    """
    print(f"----- {name_asset} daily - Close Price -----")
    print(f"Mean :\t {df['Close'].mean():.2f}")
    print(f"Std:\t{df['Close'].std() : .2f}")
    print(f"Ratio :\t {100*df['Close'].std()/df['Close'].mean():.2f}%")
    print()

def compare_btc_sp500_aapl(btc_daily) :
    """ 
    Compare daily price of Bitcoin, S&P500 and Apple.
    Args:
        btc_daily (DataFrame): Dataframe with the daily data.
    """

    basic_stats(btc_daily, "BTC")
    gspc_daily = get_daily("^GSPC", btc_daily.index.min())
    gspc_daily = gspc_daily.loc[:btc_daily.index.max()]
    basic_stats(gspc_daily, "BTC")

    aapl_daily = get_daily("AAPL", btc_daily.index.min())
    aapl_daily = aapl_daily.loc[:btc_daily.index.max()]
    basic_stats(aapl_daily, "BTC")

def powerlaw_estimation(df) :
    
    #Group data by 30 minutes
    df = df.groupby(pd.Grouper(freq='30T')).mean()
    abslogrets = np.abs(df['Logrets'])
    myfit = powerlaw.Fit(abslogrets)
    return myfit

def powerlaw_compare(fit, name) :

    distributions = ['lognormal', 'exponential']

    res = dict()
    for distrib in distributions : 
        res_test = fit.distribution_compare("power_law", distrib, normalized_ratio=True)
        res[distrib] = [res_test[0], res_test[1], (res_test[0]>0 and res_test[1]<0.05)]
    
    res_df = pd.DataFrame(res, index=['z-score', 'p-value', 'powerlaw favored ?'])
    res_df['asset'] = name
    return res_df

def powerlaw_estimation_all(df_btc, df_eth, df_ada, df_ltc) :
    cryptos = ['btc', 'ltc', 'ada', 'eth']

    fitted = dict()
    distrib_df = []
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
        
