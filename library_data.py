import pandas as pd
import datatable as dt 
import numpy as np
import yfinance as yf
import glob 
import dask
import os
from datetime import datetime, date, timedelta
import time
from pytrends.request import TrendReq
from itertools import combinations

def get_daily(name, start) :
    """
    Get historical daily price from Yahoo Finance API.
    Args:
        start (str): Start date (format %Y-%m-%d).
    Returns:
        df (DataFrame): Dataframe with the historical data.
    """ 
    ticker = yf.Ticker(name)
    df = ticker.history(start=start)
    return df[['Close', 'Volume']]

def get_btc_intraday(path_file) :
    """
    Get historical intraday Bitcoin's price from csv file.
    Args:
        path_file (str): Path to the csv file.
    Returns:
        df (DataFrame): Dataframe with the historical data.
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


def intersection_timeframes(btc_daily, btc_intraday) :
    """
    Get the new timeframe as the intersection of the indexes of the two datframes. 
    Args:
        btc_daily (DataFrame): Dataframe with the daily data.
        btc_intraday (DataFrame): Dataframe with the intraday data.
    Returns:
        btc_daily, btc_intraday: Dataframes with the new timeframe as index.
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
    """
    Set new timeframe for a given dataframe.
    Args:
        start_date (str): Start date (format %Y-%m-%d %H:%m:%s).
        end_date (str): End date (format %Y-%m-%d %H:%m:%s).
    Returns:
        df (DataFrame): Dataframe with the new timeframe as index.
    """
    df = df.loc[df.index <= end_date]
    df = df.loc[df.index >= start_date]
    return df


@dask.delayed
def close_price(path_file, save_dir):

    df = pd.read_parquet(path_file)
    try : 
        df = df[['time_exchange', 'asks', 'bids']]
    except KeyError :
        print(path_file)
        return None

    df['time_exchange'] = pd.to_datetime(df['time_exchange'])
    #Extract hour:mm
    df['minutes'] = df['time_exchange'].dt.hour*60 + df['time_exchange'].dt.minute
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
    path_files = glob.glob(path_dir + "*")
    save_dir = "data/raw/intraday_historical/" + name_asset + "/"
    allpromises = [close_price(path_file, save_dir) for path_file in path_files]
    alldata = dask.compute(allpromises)[0]

@dask.delayed
def open_file(path) : 
    df = pd.read_parquet(path)
    return df

def format_datetime(x): 
    return x.strftime("%Y-%m-%d %H:%M:%S")

def merge_historical_data() : 
    name_assets = ["ADA", "LTC", "ETH"]

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
        data.index = [format_datetime(x) for x in data.index]
        data.index = pd.to_datetime(data.index)
        data = data.rename_axis("Date")

        #Save merged file 
        data.to_parquet(f"data/raw/intraday_historical/{name_asset}_intraday",
                        use_deprecated_int96_timestamps=True, compression="brotli")

# ---------- Google Trends ----------
def create_timeframes() :
    """
    Create a list of timeframes to search
    Args :
        start_date : datetime object of the start date
        end_date : datetime object of the end date
    Returns :
        timeframes : list of timeframes
    """
    timeframes = []

    for month in range(1,12) :
        f_month_start = str(month).zfill(2)
        f_month_end = str(month + 1).zfill(2)
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
    """
    Normalize data based on values of previous timeframe.
    Args :
        previous_day : dataframe of the previous timeframe
        current_day : dataframe of the current timeframe
        col : string of the column to normalize
    Returns :
        current_day : dataframe of the current timeframe normalized
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
    """
    API call to Google Trends for a single timeframe
    Args :
        kw_list : list of keywords to search
        timeframe : string of the timeframe to search
    Returns :
        df : dataframe of the results, one column per keyword
    """
    pytrends = TrendReq(hl='en-US', tz=360)
    pytrends.build_payload(kw_list=kw_list, timeframe=timeframe)
    df = pytrends.interest_over_time()
    df.drop(columns=['isPartial'], inplace=True)
    return df


def get_google_trends_data(kw_list, timeframes) :
    """
    API call to Google Trends for a list of timeframes
    Args :
        kw_list : list of keywords to search
        timeframes : list of timeframes to search
    Returns :
        df : dataframe of the results
    """
    #Make API requests on each timeframe
    previous_day = get_google_trends_data_one_timeframe(kw_list, timeframes[0])
    for index, timeframe in enumerate(timeframes[1:]) :
        time.sleep(2)
        #API call
        current_day = get_google_trends_data_one_timeframe(kw_list, timeframe)
        #Normalize data for each column
        for col in current_day.columns :
            current_day = normalize_data(previous_day, current_day, col)
        #Concatenate dataframes
        previous_day = pd.concat([previous_day, current_day])

    return previous_day

def main_google_trends_daily(index_kw) :
    """
    Main function to collect data from Google Trends on a daily basis
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


def main_google_trends_hourly(kw_index) : 
    """ 
    Main function to get data from Google Trends on a hourly basis
    """
    data_dir = "data/raw/google_trends/hourly/downloaded/"

    previous_day = pd.read_csv(f"{data_dir}{kw_index}_1.csv", skiprows=1, index_col=0)
    previous_day = previous_day.apply(pd.to_numeric, errors='coerce')

    for idx in range(2,62) : 
        current_day = pd.read_csv(f"{data_dir}{kw_index}_{idx}.csv", skiprows=1, index_col=0)
        current_day = current_day.apply(pd.to_numeric, errors='coerce')
        
        #Normalize each column
        for col in current_day.columns :
            current_day = normalize_data(previous_day, current_day, col)
        
        #Concatenate dataframes
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




def merge_google_trends() :
    """
    Merge all csv files into one
    """
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

def select_keywords(df) : 
    """
    Select keywords with max index >= 50 
    """
    #Rank keywords by max value of interest index
    df_max = df.max(axis=0).sort_values(ascending=False)
    #Columns to remove
    to_remove = df_max[df_max < 50].index
    
    df = df.copy()
    df.drop(columns=to_remove, axis=1, inplace=True)
    return df 

def bigrams_keywords(kw_list) : 
    """
    Create bigrams of keywords.
    """
    all_combinations = []
    for r in range(1, 3):
        all_combinations.extend(list(comb) for comb in combinations(kw_list, r))
    return all_combinations

def index_bigrams(kw_list, df) : 

    for kw in kw_list : 
        if len(kw) == 2 : 
            name_col = "_".join(kw)
            df[name_col] = (df[kw[0]] + df[kw[1]])/2
    
    return df

def minutes2hours(df) :
    """ 
    Group intraday data by hour
    """
    df['date_hour'] = pd.to_datetime(df.index.strftime("%Y-%m-%d %H"))
    df = df.groupby(['date_hour']).mean()
    df = df.rename_axis("Date")
    return df

def proprocessing_keywords(df, start_date, end_date) :
    df.index = pd.to_datetime(df.index)
    df = df.rename_axis("Date")
    df = filter_timeframes(start_date, end_date, df)

    df.reset_index(inplace=True)
    df = df.drop_duplicates(subset="Date").set_index("Date")

    return df

def data_transfer_entropy(df_btc, df_ada, df_eth, df_ltc, kw_df) :
    df = pd.concat([df_btc, df_ada, df_eth, df_ltc], join="inner", axis=1)
    df.columns = ["BTC_Logrets", "ADA_Logrets", "ETH_Logrets", "LTC_Logrets"]
    df = pd.concat([df, kw_df], join="inner", axis=1)
    df = df.fillna(0)
    #Save 
    to_save = "data/clean/googleTrends_btc_ada_eth_ltc"
    df.to_parquet(to_save, use_deprecated_int96_timestamps=True, compression="brotli")
