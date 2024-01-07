import pandas as pd
import datatable as dt 

import numpy as np
import yfinance as yf

import glob 
import dask

from itertools import combinations
import os
from datetime import datetime, date, timedelta
import time

from pytrends.request import TrendReq

# ---------- Timeframes management ----------

def intersection_timeframes(btc_daily, btc_intraday) :
    """Get intersection of timeframes between daily and intraday data.

    :param btc_daily: Dataframe with daily data.
    :type btc_daily: DataFrame
    :param btc_intraday: Dataframe with intraday data.
    :type btc_intraday: DataFrame
    :return: Dataframes with the intersection of timeframes.
    :rtype: DataFrame
    """
    #Extract Day from index 
    btc_daily['Day'] = btc_daily.index.date
    btc_intraday['Day'] = btc_intraday.index.date

    #Get intersection of days
    btc_daily_index = set(btc_daily['Day'].tolist())
    btc_intraday_index = set(btc_intraday['Day'].tolist())
    btc_index = btc_daily_index.intersection(btc_intraday_index)

    #Filter datasets
    btc_daily = btc_daily.loc[btc_daily["Day"].isin(btc_index)]
    btc_intraday = btc_intraday.loc[btc_intraday["Day"].isin(btc_index)]

    #Drop Day column
    btc_daily.drop(columns=['Day'], inplace=True)
    btc_intraday.drop(columns=['Day'], inplace=True)

    return btc_daily, btc_intraday

def filter_timeframes(start_date, end_date, df) : 
    """Filter dataframe by start and end date.

    :param start_date: Start date.
    :type start_date: Datetime object
    :param end_date: End date.
    :type end_date: Datetime object
    :param df: Dataframe to filter.
    :type df: DataFrame
    :return: Filtered dataframe.
    :rtype: DataFrame
    """
    df = df.loc[df.index <= end_date]
    df = df.loc[df.index >= start_date]
    return df

def format_datetime(x): 
    """Format datetime object to string.

    :param x: Datetime object.
    :type x: Datetime object
    :return: String of the datetime object.
    :rtype: String
    """
    return x.strftime("%Y-%m-%d %H:%M:%S")

def minutes2hours(df) :
    """Convert dataframe with minute frequency to dataframe with hour frequency.

    :param df: Dataframe with minute frequency.
    :type df: DataFrame
    :return: Dataframe with hour frequency.
    :rtype: DataFrame
    """
    df['date_hour'] = pd.to_datetime(df.index.strftime("%Y-%m-%d %H"))
    df = df.groupby(['date_hour']).mean()
    df = df.rename_axis("Date")
    return df

# ---------- Historical data ----------

@dask.delayed
def open_file(path) : 
    """Open parquet file. Function to use with dask.

    :param path: Path to the file.
    :type path: String
    :return: Dataframe.
    :rtype: DataFrame
    """
    df = pd.read_parquet(path)
    return df

def get_daily(name, start) :
    """Get historical daily price from Yahoo Finance API.

    :param name: Name of the asset.
    :type name: String
    :param start: Start date (format %Y-%m-%d).
    :type start: String
    :return: Dataframe with the historical data (Close and Volume prices)
    :rtype: DataFrame
    """
    ticker = yf.Ticker(name)
    df = ticker.history(start=start)
    return df[['Close', 'Volume']]

def get_btc_intraday(path_file) :
    """Get historical intraday Bitcoin's price from csv file.

    :param path_file: Path to the csv file.
    :type path_file: String
    :return: Dataframe with the historical data (Close and Volume prices)
    :rtype: DataFrame
    """
    
    df = dt.fread(path_file)

    df.names = [name.capitalize() for name in df.names]
    df = df[::-1,:]         #Reverse row order
    df = df.to_pandas() 

    df.rename({'Timestamp': 'Date'}, axis=1, inplace=True)
    df.set_index('Date', inplace=True)   

    df = df[['Close', 'Volume']]

    if df.isnull().values.any() :
        print('There are NaN values in the dataframe')
    else :
        print('There are no NaN values in the dataframe')

    df.dropna(inplace=True)
    return df

@dask.delayed
def close_price(path_file, save_dir):
    """Extract close price from trade data for one file. 

    :param path_file: Path to the trade data.
    :type path_file: String
    :param save_dir: Directory to save the file.
    :type save_dir: String
    :return: Dataframe with the close price.
    :rtype: DataFrame
    """
    #Open file
    df = pd.read_parquet(path_file)
    try : 
        df = df[['time_exchange', 'asks', 'bids']]
    except KeyError :
        print(path_file)
        return None
    
    #Extract hour:mm
    df['time_exchange'] = pd.to_datetime(df['time_exchange'])
    df['minutes'] = df['time_exchange'].dt.hour*60 + df['time_exchange'].dt.minute

    #Extract close price : mid price between max bid and min ask for the last trade for each minute
    grouped = df.groupby('minutes').last().set_index('time_exchange')
    grouped['max_bid'] = grouped['bids'].apply(lambda x : max(x, key=lambda y: y['price'])["price"])
    grouped['min_ask'] = grouped['asks'].apply(lambda x : min(x, key=lambda y: y['price'])["price"])
    grouped['mid'] = (grouped['max_bid'] + grouped['min_ask'])/2
    grouped = grouped[['mid']]

    #Save file 
    if not os.path.exists(save_dir) :
        os.mkdir(save_dir)
    
    name_file = path_file.split("\\")[-1]
    grouped.to_parquet(save_dir+name_file, use_deprecated_int96_timestamps=True, compression="brotli")

def close_price_all(path_dir, name_asset) : 
    """Extract close price from trade data for all files in a directory.

    :param path_dir: Path to the directory.
    :type path_dir: String
    :param name_asset: Name of the asset.
    :type name_asset: String
    """
    path_files = glob.glob(path_dir + "*")
    save_dir = "data/raw/intraday_historical/" + name_asset + "/"
    allpromises = [close_price(path_file, save_dir) for path_file in path_files]
    alldata = dask.compute(allpromises)[0]

def merge_historical_data() : 
    """Merge historical data for ADA, LTC, ETH (one file per asset).

    :return: Dataframe with the historical data.
    :rtype: DataFrame
    """

    name_assets = ["ADA", "LTC", "ETH"]

    #For each asset
    for name_asset in name_assets : 
        dir_path = f"data/raw/intraday_historical/{name_asset}/*"

        #Open files
        allpaths = glob.glob(dir_path)
        allpromises = [open_file(path) for path in allpaths]
        alldata = dask.compute(allpromises)[0]

        #Merge files
        data = pd.concat(alldata)
        data = data.sort_index()

        #Change column names
        data.columns = ["Close"] 

        #Change index
        data.index = [format_datetime(x) for x in data.index]
        data.index = pd.to_datetime(data.index)
        data = data.rename_axis("Date")

        #Save merged file 
        data.to_parquet(f"data/raw/intraday_historical/{name_asset}_intraday",
                        use_deprecated_int96_timestamps=True, compression="brotli")

# ---------- Google Trends ----------
def create_timeframes() :
    """Create list of timeframes to collect data from Google Trends (timeframe format : "YYYY-MM-DD YYYY-MM-DD").

    :return: List of timeframes.
    :rtype: List
    """
    timeframes = []

    for month in range(1,12) :
        #Add 0 if month < 10
        f_month_start = str(month).zfill(2)
        f_month_end = str(month + 1).zfill(2)

        #Create timeframe
        start = f"2021-{f_month_start}-01"
        if month+1 in [1,3,5,7,8,10,12] :
          end = f"2021-{f_month_end}-31"
        elif month+1 in [4,6,9,11] :
          end = f"2021-{f_month_end}-30"
        else :
          end = f"2021-{f_month_end}-28"

        timeframes.append(start + ' ' + end)

    return timeframes

def normalize_data(previous_day, current_day, col) :
    """Normalize Google Trends indexes of "current_day" dataframe using data from "previous_day" dataframe.

    :param previous_day: Dataframe with Google Trends indexes from previous day.
    :type previous_day: DataFrame
    :param current_day: Dataframe with Google Trends indexes from current day.
    :type current_day: DataFrame
    :param col: Column name, containing Google Trends indexes for a specific keyword.
    :type col: String
    :return: Current_day dataframe with normalized Google Trends indexes.
    :rtype: DataFrame
    """
    #First value of current day
    current_day_value = current_day.iloc[0,:][col]

    #Value of previous dataframe at same time
    previous_day_value = previous_day.loc[current_day.index[0],:][col]

    #One of the two values is null, no normalization
    if current_day_value == 0 or previous_day_value == 0 : 
        current_day_value = 1 
        previous_day_value = 1

    #Normalize all values of current day
    current_day[col] = current_day[col] * (previous_day_value / current_day_value)

    return current_day

def get_google_trends_data_one_timeframe(kw_list, timeframe) :
    """API call to Google Trends for one timeframe.

    :param kw_list: List of keywords to search.
    :type kw_list: List
    :param timeframe: Timeframe to search (format : "YYYY-MM-DD YYYY-MM-DD").
    :type timeframe: String
    :return: Dataframe with the results.
    :rtype: DataFrame
    """
    pytrends = TrendReq(hl='en-US', tz=360)
    pytrends.build_payload(kw_list=kw_list, timeframe=timeframe)
    df = pytrends.interest_over_time()
    df.drop(columns=['isPartial'], inplace=True)
    return df


def get_google_trends_data(kw_list, timeframes) :
    """API call to Google Trends for a list of timeframes.

    :param kw_list: List of keywords to search.
    :type kw_list: List
    :param timeframes: List of timeframes to search (format : "YYYY-MM-DD YYYY-MM-DD").
    :type timeframes: List
    :return: Dataframe with the results.
    :rtype: DataFrame
    """
    #Retrieve previous day data
    previous_day = get_google_trends_data_one_timeframe(kw_list, timeframes[0])

    #For each timeframe
    for index, timeframe in enumerate(timeframes[1:]) :
        #Retrieve current day data
        current_day = get_google_trends_data_one_timeframe(kw_list, timeframe)
        time.sleep(2)

        #Normalize current day data
        for col in current_day.columns :
            current_day = normalize_data(previous_day, current_day, col)

        #Concatenate previous and current day data 
        previous_day = pd.concat([previous_day, current_day])

    return previous_day

def main_google_trends_daily(index_kw) :
    """Main function to make API calls to Google Trends to retrieve daily data. 

    :param index_kw: Index of the list of keywords to search.
    :type index_kw: Int
    """
    #Create list of timeframes
    timeframes = create_timeframes()

    #List of keywords to search
    kw_list =   [   [ "Bitcoin", "BTC", "Ethereum", "ETH"],
                    ["Cardano", "ADA", "Litecoin", "LTC"],
                    ["crypto", "cryptocurrency", "trading", "Binance"],
                    ["Bitstamp", "China", "FED", "Musk"]
                ]

    #Collect data for each list of keywords
    print("Collecting data for keywords : ", kw_list[index_kw])
    data = get_google_trends_data(kw_list[index_kw], timeframes)

    #Save data 
    if not os.path.exists("data/raw/google_trends/daily/") :
        os.mkdir("data/raw/google_trends/daily/")
    
    data.to_csv("data/raw/google_trends/daily/" + "_".join(kw_list[index_kw]) + ".csv")


def google_trends_hourly(kw_index) : 
    """Normalize hourly Google Trends data and merge results into one file.

    :param kw_index: Index of the list of keywords to search.
    :type kw_index: Int
    """

    data_dir = "data/raw/google_trends/hourly/downloaded/"

    #Retrieve previous day data
    previous_day = pd.read_csv(f"{data_dir}{kw_index}_1.csv", skiprows=1, index_col=0)
    previous_day = previous_day.apply(pd.to_numeric, errors='coerce')

    #Normalize each column
    for idx in range(2,62) : 
        #Retrieve current day data
        current_day = pd.read_csv(f"{data_dir}{kw_index}_{idx}.csv", skiprows=1, index_col=0)
        current_day = current_day.apply(pd.to_numeric, errors='coerce')
        
        #Normalize current day data
        for col in current_day.columns :
            current_day = normalize_data(previous_day, current_day, col)
        
        #Concatenate previous and current day data 
        previous_day = pd.concat([previous_day, current_day])

    #Change column names 
    substring_to_remove = ": (États-Unis)"
    previous_day.columns = previous_day.columns.str.replace(substring_to_remove, '')
    
    #And index 
    previous_day = previous_day.rename_axis("date")

    #Save 
    if not os.path.exists("data/raw/google_trends/hourly/merged/") :
        os.mkdir("data/raw/google_trends/hourly/merged/")
    
    to_save = "data/raw/google_trends/hourly/merged/" + "_".join(previous_day.columns)
    previous_day.to_parquet(to_save, use_deprecated_int96_timestamps=True, compression="brotli")


def main_google_trends_hourly() :
    """Main function to normalize hourly Google Trends data and merge results into one file.
    """
    #Normalize data for each list of keywords
    for kw_index in range(4) :
        google_trends_hourly(kw_index)

    #Merge files
    path_files = glob.glob("data/raw/google_trends/hourly/merged/*")
    path_all = "data/raw/google_trends/hourly/merged\\all"
    if path_all in path_files : 
        path_files.remove(path_all)
    
    df = pd.read_parquet(path_files[0])
    for path_file in path_files[1:] :
        df_tmp = pd.read_parquet(path_file)
        df = df.merge(df_tmp, on="date", how="inner")
    
    df.to_parquet("data/raw/google_trends/hourly/merged/all", 
                  use_deprecated_int96_timestamps=True, compression="brotli")

def proprocessing_keywords(df, start_date, end_date) :
    """Preprocess Google Trends data : filter by start and end date, drop duplicates, set index.

    :param df: Dataframe with Google Trends data.
    :type df: DataFrame
    :param start_date: Start date.
    :type start_date: Datetime object
    :param end_date: End date.
    :type end_date: Datetime object
    :return: Preprocessed dataframe.
    :rtype: DataFrame
    """
    df.index = pd.to_datetime(df.index)
    df = df.rename_axis("Date")
    df = filter_timeframes(start_date, end_date, df)

    df.reset_index(inplace=True)
    df = df.drop_duplicates(subset="Date").set_index("Date")

    return df

def data_transfer_entropy(df_btc, df_ada, df_eth, df_ltc, kw_df) :
    """Prepare data for transfer entropy analysis : merge dataframes, drop duplicates, fill NaN values.
    
    :param df_btc: Dataframe with Bitcoin's historical data.
    :type df_btc: DataFrame
    :param df_ada: Dataframe with Cardano's historical data.
    :type df_ada: DataFrame
    :param df_eth: Dataframe with Ethereum's historical data.
    :type df_eth: DataFrame
    :param df_ltc: Dataframe with Litecoin's historical data.
    :type df_ltc: DataFrame
    :param kw_df: Dataframe with Google Trends data.
    :type kw_df: DataFrame
    :return: Dataframe with all the data.
    :rtype: DataFrame
    """
    #Merge crypto datasets. 
    df = pd.concat([df_btc, df_ada, df_eth, df_ltc], join="inner", axis=1)
    df.columns = ["BTC_Logrets", "ADA_Logrets", "ETH_Logrets", "LTC_Logrets"]

    #Merge crypto and Google Trends datasets.
    df = pd.concat([df, kw_df], join="inner", axis=1)
    df = df.fillna(0)
    
    #Save 
    to_save = "data/clean/google_trends/googleTrends_btc_ada_eth_ltc"
    df.to_parquet(to_save, use_deprecated_int96_timestamps=True, compression="brotli")
