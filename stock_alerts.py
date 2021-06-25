# pip install pandas-datareader
# pip install yfinance
from pandas_datareader import data as pdr
from datetime import date as datetime
import yfinance as yf
yf.pdr_override()
import pandas as pd
ticker_list=['PFE','EBAY','ONLN','SCHW','AES','COG']
today = datetime.today()
# We can get data by our choice by giving days bracket
# start_date="2021–05–01"
# end_date="2021–06–23"
start_date = datetime(2020, 5, 1)
end_date = datetime(2021, 6, 22)
files=[]
def getData(ticker):

    data = pdr.get_data_yahoo(ticker, start=start_date, end=today)
    dataname= ticker+'_'+str(today)
    files.append(dataname)
    #print(data)
    #This selects the 'Adj Close' column
    close = data['Adj Close']
    #This converts the date strings in the index into pandas datetime format:
    close.index = pd.to_datetime(close.index)
    sma50 = close.rolling(window=50).mean()
    result = sma50.get(key = '2021-06-22')
    print(ticker , '',result)
    #print(sma50)

for tik in ticker_list:
    getData(tik)