import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import plotly.graph_objects as go
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.distributions.empirical_distribution import ECDF

import dask

from library_data import *
from library_stats import *
from library_stylized_facts import *

# ----- Data Collection - Google Trends -----
def keywords_evolution(df, nb_rows, nb_cols) :
    """Plot variations in Google Trends index for a set of keywords. 

    :param df: Dataframe containing Google Trends indexes, one column corresponds to one keyword.
    :type df: DataFrame
    :param nb_rows: Number of rows for subplots. 
    :type nb_rows: Int
    :param nb_cols: Number of columns for subplots. 
    :type nb_cols: Int
    """
    fig, axs = plt.subplots(nb_rows, nb_cols, figsize=(10, 8), sharex=True)

    for i, ax in enumerate(axs.flatten()) :
        ax.plot(df.iloc[:,i])
        ax.set_title(df.columns[i], fontdict={'fontsize': 16})
        
    fig.tight_layout()

    #Save figure
    if not os.path.exists("figures/") :
        os.mkdir("figures/")
    plt.savefig("figures/keywords_evolution.png")

    plt.show()

def rank_keywords_index(df) :
    """Plot barplot displaying ranking of keywords based on their averaged Google Trends index over the period of interest. 

    :param df: Dataframe containing Google Trends indexes, one column corresponds to one keyword.
    :type df: DataFrame
    """
    #Rank keywords by max value of interest index
    df_max = df.mean(axis=0).sort_values(ascending=False)

    sns.set_theme(style="whitegrid")
    sns.set(rc={'figure.figsize':(11.5,6)})
    ax = sns.barplot(x=df_max.values, y=df_max.index, orient="h")
    ax.set(ylabel=None)

    #Save figure
    if not os.path.exists("figures/") :
        os.mkdir("figures/")
    plt.savefig("figures/ranking_keywords_index.png")

    plt.show()

# ----- In-Depth Analysis of Bitcoin -----
def compare_daily_intraday(column_name, df_daily, df_intraday) :
    """
    Plot variations of "column_name" for both daily and intraday data of 2 different dataframes. 

    :param column_name: Column name to plot ("Close" or "Logrets")
    :type column_name: String
    :param df_daily: Dataframe containing daily data for a specific asset
    :type df_daily: DataFrame
    :param df_intraday: Dataframe containing daily data for a specific asset
    :type df_intraday: DataFrame
    """
    fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    
    #Price Daily
    ax[0].plot(df_daily.index, df_daily[column_name])
    ax[0].set_title('DAILY')
    ax[0].set_ylabel('USD')

    #Price Intraday
    ax[1].plot(df_intraday.index, df_intraday[column_name], color='orange')
    ax[1].set_title('INTRADAY')
    ax[1].set_ylabel('USD')
    
    #Save figure
    if not os.path.exists("figures/") :
        os.mkdir("figures/")
    plt.savefig(f"figures/btc_{column_name}.png")
    
    fig.show()

def annotated_btc(df_daily) :
    """Plot variations of BTC prices, with annotations.

    :param df_daily: Dataframe containing closing price of BTC ("Close" column). 
    :type df_daily: DataFrame
    """
    shapes = [  dict(x0='2020-01-02', x1='2020-01-02', y0=0, y1=1, xref='x', yref='paper', line_width=1,  opacity=0.5),
                dict(x0='2020-03-20', x1='2020-03-20', y0=0, y1=1, xref='x', yref='paper', line_width=1, opacity=0.5),
                dict(x0='2021-04-15', x1='2021-04-15', y0=0, y1=1, xref='x', yref='paper', line_width=1, opacity=0.5),
                dict(x0='2021-11-03', x1='2021-11-03', y0=0, y1=1, xref='x', yref='paper', line_width=1, opacity=0.5)]
    
    annotations=[dict(x='2020-01-02', y=1.1, xref='x', yref='paper',showarrow=False, xanchor='left', text='😷Covid'),
                dict(x='2020-03-11', y=0.13, xref='x', yref='paper',showarrow=True, xanchor='center', text='Flash crash', font=dict(size=9)),
                dict(x='2020-08-15', y=1.1, xref='x', yref='paper',showarrow=False, xanchor='left', text='📈Bull run'),
                dict(x='2021-05-01', y=0.83, xref='x', yref='paper',showarrow=True, xanchor='center', text='China\'s warnings', font=dict(size=9)),
                dict(x='2021-06-01', y=1.1, xref='x', yref='paper',showarrow=False, xanchor='left', text='🚨Instability'),
                dict(x='2021-11-03', y=0.9, xref='x', yref='paper',showarrow=True, xanchor='center', text='Fed announcement', font=dict(size=9)),
                dict(x='2022-09-01', y=1.1, xref='x', yref='paper',showarrow=False, xanchor='center', text='📉Bear market'),
                dict(x='2022-03-01', y=0.62, xref='x', yref='paper',showarrow=True, xanchor='left', text='Raise of Interest rate', font=dict(size=9)),
                ]

    fig = go.Figure(data=[go.Scatter(x=df_daily.index, y=df_daily['Close'])])
    fig.update_layout(annotations=annotations, shapes=shapes)

    if not os.path.exists("figures/") :
        os.mkdir("figures/")
    fig.write_image("figures/btc_analysis.png")

    fig.show()

def annotated_btc_sp500(btc_daily, gspc_daily) :
    """Plot variations of BTC and S&P 500 prices, with annotations.

    :param btc_daily: Dataframe containing closing price of BTC ("Close" column). 
    :type btc_daily: DataFrame
    :param gspc_daily: Dataframe containing closing price of S&P 500 ("Close" column). 
    :type gspc_daily: DataFrame
    """
    #Normalize
    btc_daily['Close_norm'] = (btc_daily['Close']-btc_daily['Close'].mean())/btc_daily['Close'].std()
    gspc_daily['Close_norm'] = (gspc_daily['Close']-gspc_daily['Close'].mean())/gspc_daily['Close'].std()

    #Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=btc_daily.index, y=btc_daily['Close_norm'], name='BTC'))
    fig.add_trace(go.Scatter(x=gspc_daily.index, y=gspc_daily['Close_norm'], name='S&P 500'))

    shapes = [dict(x0='2020-01-02', x1='2020-01-02', y0=0, y1=1, xref='x', yref='paper', line_width=1,  opacity=0.5),
            dict(x0='2020-03-20', x1='2020-03-20', y0=0, y1=1, xref='x', yref='paper', line_width=1, opacity=0.5),
            dict(x0='2021-04-15', x1='2021-04-15', y0=0, y1=1, xref='x', yref='paper', line_width=1, opacity=0.5),
            dict(x0='2021-11-03', x1='2021-11-03', y0=0, y1=1, xref='x', yref='paper', line_width=1, opacity=0.5)]
    
    annotations=[dict(x='2020-01-02', y=1.1, xref='x', yref='paper',showarrow=False, xanchor='left', text='😷Covid'),
                dict(x='2020-03-11', y=0.3, xref='x', yref='paper',showarrow=True, xanchor='center', text='Flash crash', font=dict(size=9)),
                dict(x='2020-08-15', y=1.1, xref='x', yref='paper',showarrow=False, xanchor='left', text='📈Bull run'),
                dict(x='2021-06-01', y=1.1, xref='x', yref='paper',showarrow=False, xanchor='left', text='🚨Instability'),
                dict(x='2021-05-01', y=0.85, xref='x', yref='paper',showarrow=True, xanchor='center', text='China\'s warnings', font=dict(size=9)),
                dict(x='2022-09-01', y=1.1, xref='x', yref='paper',showarrow=False, xanchor='center', text='📉Bear market'),
                dict(x='2021-11-03', y=0.9, xref='x', yref='paper',showarrow=True, xanchor='center', text='Fed announcement', font=dict(size=9)),
                dict(x='2022-03-01', y=0.7, xref='x', yref='paper',showarrow=True, xanchor='left', text='Raise of Interest rate', font=dict(size=9)),
                ]

    fig.update_layout(annotations=annotations, shapes=shapes)
    
    if not os.path.exists("figures/") :
        os.mkdir("figures/")
    fig.write_image("figures/btc_sp500.png")

    fig.show()

# ----- Stylized facts on Price Movements -----
def logreturns(btc_df, eth_df, ada_df, ltc_df) :
    """Plot logreturns of ADA, BTC, ETH, LTC.

    :param btc_df: Dataframe containing closing price of BTC ("Logrets" column). 
    :type btc_df: DataFrame
    :param eth_df: Dataframe containing closing price of ETH ("Logrets" column). 
    :type eth_df: DataFrame
    :param ada_df: Dataframe containing closing price of ADA ("Logrets" column). 
    :type ada_df: DataFrame
    :param ltc_df: Dataframe containing closing price of LTC ("Logrets" column). 
    :type ltc_df: DataFrame
    """
    fig, ax = plt.subplots(4, 1, figsize=(6, 8), sharex=True)

    ax[0].plot(btc_df.index, btc_df['Logrets'], color='blue')
    ax[0].set_title('BTC')

    ax[1].plot(eth_df.index, eth_df['Logrets'], color='red')
    ax[1].set_title('ETH')

    ax[2].plot(ada_df.index, ada_df['Logrets'], color='green')
    ax[2].set_title('ADA')

    ax[3].plot(ltc_df.index, ltc_df['Logrets'], color='black')
    ax[3].set_title('LTC')

    #Save figure
    if not os.path.exists("figures/") :
        os.mkdir("figures/")
    plt.savefig("figures/logreturns.png")

    fig.show()

def acf(btc_df, eth_df, ada_df, ltc_df, col_name) :
    """Plot ACF for "col_name" data of ADA, BTC, ETH, LTC.

    :param btc_df: Dataframe containing closing price of BTC ("Logrets" column). 
    :type btc_df: DataFrame
    :param eth_df: Dataframe containing closing price of ETH ("Logrets" column). 
    :type eth_df: DataFrame
    :param ada_df: Dataframe containing closing price of ADA ("Logrets" column). 
    :type ada_df: DataFrame
    :param ltc_df: Dataframe containing closing price of LTC ("Logrets" column). 
    :type ltc_df: DataFrame
    :param col_name: Name of the column for which we want to plot the ACF.
    :type col_name: String
    """
    
    #Define lags
    lags = [lag for lag in range(0, 41)]

    #Compute autocorrelation using Dask
    btc_acf = autocorrelation_dask(btc_df, col_name, lags)
    eth_acf = autocorrelation_dask(eth_df, col_name, lags)
    ada_acf = autocorrelation_dask(ada_df, col_name, lags)
    ltc_acf = autocorrelation_dask(ltc_df, col_name, lags)

    fig, ax = plt.subplots(4, 1, figsize=(4,8), sharex=True, sharey=True)

    ax[0].stem(lags, btc_acf, linefmt='b-', markerfmt='bo', basefmt='b-')
    ax[0].set_title('BTC')

    ax[1].stem(lags, eth_acf, markerfmt='ro', basefmt='r-', linefmt='r-')
    ax[1].set_title('ETH')

    ax[2].stem(lags, ada_acf, markerfmt='go', basefmt='g-', linefmt='g-')
    ax[2].set_title('ADA')

    ax[3].stem(lags, ltc_acf, markerfmt='ko', basefmt='k-', linefmt='k-')
    ax[3].set_title('LTC')
    ax[3].set_xlabel('Lag')

    #Save figure
    if not os.path.exists("figures/") :
        os.mkdir("figures/")
    plt.savefig(f"figures/acf_{col_name}.png")

    plt.show()

def acf_volatility(btc_df, eth_df, ada_df, ltc_df, time_window) : 
    """Plot ACF for Volatility of ADA, BTC, ETH, LTC.

    :param btc_df: Dataframe containing closing price of BTC ("Logrets" column). 
    :type btc_df: DataFrame
    :param eth_df: Dataframe containing closing price of ETH ("Logrets" column). 
    :type eth_df: DataFrame
    :param ada_df: Dataframe containing closing price of ADA ("Logrets" column). 
    :type ada_df: DataFrame
    :param ltc_df: Dataframe containing closing price of LTC ("Logrets" column). 
    :type ltc_df: DataFrame
    :param time_window: Time-window for computing Volatility. 
    :type time_window: Int
    """
    #Compute volatility
    btc_vol = volatility(btc_df, time_window)
    eth_vol = volatility(eth_df, time_window)
    ada_vol = volatility(ada_df, time_window)
    ltc_vol = volatility(ltc_df, time_window)

    acf(btc_vol, eth_vol, ada_vol, ltc_vol, 'Volatility')

def complementary_ecdf(df, asset_name, ax, color) :
    """Plot complementary ECDF of log-returns ("Logrets" column) in log-log scale. 

    :param df: Dataframe containing log-returns for the asset of interest. 
    :type df: DataFrame
    :param asset_name: Asset name for plot title. 
    :type asset_name: String
    :param ax: Ax index for plotting. 
    :type ax: Int
    :param color: Color for plotting. 
    :type color: String
    :return: Subplot to add to main figure. 
    :rtype: Ax index for plotting. 
    """
    
    #Compute ECDF
    ecdf = ECDF(np.abs(df['Logrets']))

    ax.plot(ecdf.x, 1 - ecdf.y, color = color)

    #Set log-log scale
    ax.set_xscale("log")
    ax.set_yscale("log")

    #Set labels
    ax.set_ylabel('$log P_>(|r|)$')
    if asset_name == 'LTC' :
        ax.set_xlabel("log |r|")
    ax.set_title(asset_name)

    return ax

def complementary_ecdf_all(btc_df, eth_df, ada_df, ltc_df) :
    """Plot complementary ECDF of log-returns for BTC, ADA, LTC, ETH in log-log scale.

    :param btc_df: Dataframe containing log-returns for BTC. 
    :type btc_df: DataFrame
    :param eth_df: Dataframe containing log-returns for ETH. 
    :type eth_df: DataFrame
    :param ada_df: Dataframe containing log-returns for ADA. 
    :type ada_df: DataFrame
    :param ltc_df: Dataframe containing log-returns for LTC. 
    :type ltc_df: DataFrame
    """
    fig, ax = plt.subplots(4, 1, figsize=(4, 8), sharex=True, sharey=True)
    
    ax[0] = complementary_ecdf(btc_df, 'BTC', ax[0], 'blue')
    ax[1] = complementary_ecdf(eth_df, 'ETH', ax[1], 'red')
    ax[2] = complementary_ecdf(ada_df, 'ADA', ax[2], 'green')
    ax[3] = complementary_ecdf(ltc_df, 'LTC', ax[3], 'black')
    
    #Save figure
    if not os.path.exists("figures/") :
        os.mkdir("figures/")
    plt.savefig("figures/complementary_ecdf.png")

    plt.show()

def plot_volatility(btc_df, eth_df, ada_df, ltc_df, time_window) :
    """Compute and plot Volatility of Logreturns for ADA, ETH, BTC, LTC. 

    :param btc_df: Dataframe containing log-returns for BTC. 
    :type btc_df: DataFrame
    :param eth_df: Dataframe containing log-returns for ETH. 
    :type eth_df: DataFrame
    :param ada_df: Dataframe containing log-returns for ADA. 
    :type ada_df: DataFrame
    :param ltc_df: Dataframe containing log-returns for LTC. 
    :type ltc_df: DataFrame
    :param time_window: Time-window for computing Volatility. 
    :type time_window: Int
    """
    #Compute volatility and store as a Dataframe. 
    btc_vol = volatility(btc_df, time_window)
    eth_vol = volatility(eth_df, time_window)
    ada_vol = volatility(ada_df, time_window)
    ltc_vol = volatility(ltc_df, time_window)

    fig, ax = plt.subplots(4, 1, figsize=(6, 8), sharex=True)

    ax[0].plot(btc_vol.index, btc_vol['Volatility'], color='blue')
    ax[0].set_title('BTC')

    ax[1].plot(eth_vol.index, eth_vol['Volatility'], color='red')
    ax[1].set_title('ETH')

    ax[2].plot(ada_vol.index, ada_vol['Volatility'], color='green')
    ax[2].set_title('ADA')

    ax[3].plot(ltc_vol.index, ltc_vol['Volatility'], color='black')
    ax[3].set_title('LTC')

    #Save figure
    if not os.path.exists("figures/") :
        os.mkdir("figures/")
    plt.savefig("figures/volatility.png")

    plt.show()

def plot_response_plot(data_dir):
    """_summary_

    :return: _description_
    :rtype: _type_
    """

    fig, ax = plt.subplots(2,2)
    fig.supxlabel("Lag [trade counts]")
    fig.supylabel("Response")
    def plot_response_curve(data_dir,coin,max_lags=1000,date_pattern="2021-01-0[1]",ax=ax,color="black"):
        """_summary_

        :param coin: _description_
        :type coin: _type_
        :param max_lags: _description_, defaults to 1000
        :type max_lags: int, optional
        :param date_pattern: _description_, defaults to "2021-01-0[1]"
        :type date_pattern: str, optional
        :param ax: _description_, defaults to ax
        :type ax: _type_, optional
        :param color: _description_, defaults to "black"
        :type color: str, optional
        """

        def response_function(events,tau):
            response = events['s_n']*(np.log(events['midprice']) - np.log(events['midprice']).shift(tau))
            return response.mean()
        ax.set_xscale('log')
        ax.set_xlim([1,1e3])

        #Load data
        trade_files=glob.glob(data_dir + f"binanceus/binanceus_{coin}USD/trades/*_{date_pattern}")
        trade_files.sort()
        allpromises=[pd.read_parquet(fn,columns=['time_exchange','taker_side']) for fn in trade_files]
        trades=dask.compute(allpromises)[0]
        trades=pd.concat(trades)

        mob_files = glob.glob(data_dir + f"binanceus/binanceus_{coin}USD/mob/*_{date_pattern}")
        mob_files.sort()
        allpromises=[pd.read_parquet(fn,columns=['asks','bids','time_exchange']) for fn in mob_files]
        mob = dask.compute(allpromises)[0]
        mob = pd.concat(mob)

        trades['time_exchange'] = pd.to_datetime(trades['time_exchange'])
        mob['time_exchange'] = pd.to_datetime(mob['time_exchange'])
        mob['midprice'] = mob.apply(lambda row : (row['asks'][0]['price'] + row['bids'][0]['price'])/2,axis=1)
        mob['midprice_vol'] = mob.apply(lambda row : (row['asks'][0]['size'] + row['bids'][0]['size'])/2,axis=1)

        events=pd.concat([trades,mob],join="outer",keys=["time_exchange","time_exchange"]).sort_values('time_exchange').ffill()

        events['s_n'] = events.apply(lambda row : +1 if row['taker_side'] == "BUY" else -1,axis=1)

        range_x = np.arange(max_lags)
        response = [response_function(events,k) for k in range_x]
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
        ax.plot(range_x, response,label=coin,color=color)
        ax.legend(loc="best")

    plot_response_curve(data_dir,"BTC",ax=ax[0][0],color="blue")
    plot_response_curve(data_dir,"ETH",ax=ax[0][1],color="red")
    plot_response_curve(data_dir,"LTC",ax=ax[1][0],color="black")
    plot_response_curve(data_dir,"ADA",ax=ax[1][1],color="green")
    fig.tight_layout()
