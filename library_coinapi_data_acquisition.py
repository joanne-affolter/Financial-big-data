import vaex
import pandas as pd
import numpy as np
import os
from glob import glob
import requests
import pandas as pd
import os
from time import sleep

# Retrieve API key
def retrieve_api_key(name_file) : 
    api_key = None
    with open(name_file) as file:
        api_key = file.readline().strip()
    return api_key


# Retrieve historical data on Trades
def get_historical_trades(headers, index, day, save_dir, coin, exchange) :
    
    parameters = {  
        "symbol_id": f"B{exchange.upper()}_SPOT_{coin}_USD",
        "time_start" : day+'T00:00',
        "time_end" : day+'T23:59:59.999',
        "limit" : 100000, 
        "include_id" : True
    }
    r = requests.get(f"https://rest.coinapi.io/v1/trades/{exchange.upper()}_SPOT_{coin}_USD/history", 
                     headers=headers, params=parameters)

    # Check if the request was successful (status code 200)
    if r.status_code == 200:
        # Parse and print the trade data
        data = r.json()
        df = pd.DataFrame(data)
        print(f"Length of data : {len(df)}")
        #Save as csv 
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        name_file = str(index) + "_" + day
        df.to_parquet(save_dir+name_file, use_deprecated_int96_timestamps=True, compression="brotli")

    else:
        # Print an error message if the request was unsuccessful
        print(f"Error: {r.status_code} - {r.text}")

# Retrieve historical data on Trades
def get_historical_mob(headers, index, day, save_dir,coin,exchange) :
    
    parameters = {  
        "symbol_id": f"{exchange.upper()}_SPOT_{coin}_USD",
        "time_start" : day+'T00:00',
        "time_end" : day+'T23:59:59.999',
        "limit" : 90000, 
        "include_id" : True
    }
    r = requests.get(f"https://rest.coinapi.io/v1/orderbooks/{exchange.upper()}_SPOT_{coin}_USD/history", 
                     headers=headers, params=parameters)

    # Check if the request was successful (status code 200)
    if r.status_code == 200:
        # Parse and print the trade data
        data = r.json()
        df = pd.DataFrame(data)
        print(f"Length of data : {len(df)}")
        #Save as csv 
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        name_file = str(index) + "_" + day
        df.to_parquet(save_dir+name_file, use_deprecated_int96_timestamps=True, compression="brotli")

    else:
        # Print an error message if the request was unsuccessful
        print(f"Error: {r.status_code} - {r.text}")


#To call every day 
def main_get_historical_trades(path_file_API, path_file_periods, index,coin,exchange) : 
    api_key = retrieve_api_key(path_file_API)
    headers = {"X-CoinAPI-Key": api_key}

    days_list = None
    #Open the file and get the list of index to scrap (based on index)
    with open(path_file_periods) as file : 
        lines = file.readlines()
        for line in lines : 
            if line.split(":")[0].strip() == str(index) :
                days_list = line.split(":")[1][1:-1].split(" ")
    for day in days_list : 
        print(day)
        sleep(60)
        get_historical_trades(headers, index, day, save_dir = f"data/raw/bitstamp/{exchange}_{coin}USD/trades/", coin = coin, exchange = exchange)

def main_get_historical_mob(path_file_API, path_file_periods, index,coin,exchange) : 
    api_key = retrieve_api_key(path_file_API)
    headers = {"X-CoinAPI-Key": api_key}

    days_list = None
    #Open the file and get the list of index to scrap (based on index)
    with open(path_file_periods) as file : 
        lines = file.readlines()
        for line in lines : 
            if line.split(":")[0].strip() == str(index) :
                days_list = line.split(":")[1][1:-1].split(" ")

    for day in days_list : 
        sleep(30)
        print(day)
        get_historical_mob(headers, index, day, save_dir = f"data/raw/bitstamp/{exchange}_{coin}USD/mob/", coin = coin, exchange = exchange)

def check_data_integrity(coin, exchange, datatype, data_path = "/media/jprado/Elements/data/raw/"):
    paths = glob(f"{data_path}/{exchange}/{exchange}_{coin}USD/{datatype}/*")
    df = vaex.open_many(paths)
    day_counts = len(paths)
    # Check 1 : There is a single coin in the timeseries
    single_coin = len(df['symbol_id'].value_counts()) == 1
    # Check 2 : The timeseries correspond to the expected timeseries (according to foldername)
    correct_coin = coin in df['symbol_id'].value_counts().index[0]
    # Check 3 : The timeseries has all days of the year
    day_count_check = day_counts == 365
    dates = pd.to_datetime(df['time_exchange'].to_pandas_series())

    # Aggregate by day and filter for the days where the last time is before 23:58
    result = (
        dates.groupby(dates.dt.date)
        .filter(lambda x: x.max().time() < pd.to_datetime('23:58').time())
        .groupby(dates.dt.date)
        .agg(['count','max'])
    )

    return (single_coin and correct_coin and day_count_check), day_counts


def generate_returns_df(coin, exchange, datatype,raw_data_path, output_dir):
    paths = sorted(glob(f"{raw_data_path}/{exchange}/{exchange}_{coin}USD/{datatype}/*"))
    for path in paths:
        df = pd.read_parquet(path).sort_values('time_exchange')

        output_dir = f"{output_dir}/{exchange}/{exchange}_{coin}USD/{datatype}/"
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        df['log_returns'] = (df['price'] / df['price'].shift(1)).apply(np.log)
        df.drop(columns=['symbol_id', 'time_coinapi', 'uuid', 'size', 'taker_side','price']
,inplace=True) 

        # Use the export method to save each chunk to a Parquet file
        df.to_parquet(output_dir + f'log_returns_{path.split("/")[-1].split(".")[0]}.parquet')
    return df