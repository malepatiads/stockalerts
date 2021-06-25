# pip install pandas-datareader
# pip install yfinance
# pip install openpyxl

from datetime import date as datetime

import yfinance as yf
from pandas_datareader import data as pdr

yf.pdr_override()
import pandas as pd
import matplotlib.pyplot as plot
import numpy as np
from scipy.interpolate import interp1d
import seaborn as sns

def process_tickers(ticker, start, end):
    data = pdr.get_data_yahoo(ticker, start=start, end=end)  # interval="5d"
    close = data['Adj Close']
    if ticker == 'SPY':
        rdf['DATE'] = pd.to_datetime(close.index)
        rdf.index = close.index
        rdf[ticker + '_' + 'PRICE'] = round(close, 2)
    else:
        rdf[ticker + '_' + 'PRICE'] = round(close, 2)
        sma50 = close.rolling(window=50).mean()
        std50 = close.rolling(window=50).std()
        rdf[ticker + '_' + 'SMA50'] = round(sma50, 2)

        rdf[ticker + '_' + 'PR'] = round(100 * rdf[ticker + '_' + 'PRICE'] / rdf['SPY_PRICE'], 2)
        pr = rdf[ticker + '_' + 'PR']
        prsma50 = pr.rolling(window=50).mean()
        prstd50 = pr.rolling(window=50).std()
        rdf[ticker + '_PR_' + 'SMA50'] = round(prsma50, 2)
        rdf[ticker + '_PR_' + 'STD50'] = round(prstd50, 2)
        rdf[ticker + '_' + 'RS'] = round(
            100 + ((rdf[ticker + '_' + 'PR'] - rdf[ticker + '_PR_' + 'SMA50']) / rdf[ticker + '_PR_' + 'STD50']) + 1, 2)

        rdf[ticker + '_' + 'ROC'] = rdf[ticker + '_' + 'RS'].rolling(window=2).apply(
            lambda x: (100 * ((x.iloc[1] / x.iloc[0]) - 1)))
        roc = rdf[ticker + '_' + 'ROC']
        rocsma50 = roc.rolling(window=50).mean()
        rocstd50 = roc.rolling(window=50).std()
        rdf[ticker + '_ROC_' + 'SMA50'] = round(rocsma50, 2)
        rdf[ticker + '_ROC_' + 'STD50'] = round(rocstd50, 2)
        rdf[ticker + '_' + 'RM'] = round(
            100 + ((rdf[ticker + '_' + 'ROC'] - rdf[ticker + '_ROC_' + 'SMA50']) / rdf[ticker + '_ROC_' + 'STD50']) + 1,
            2)


def draw_plot(ticker):
    return rdf.plot(x=ticker + '_' + 'RS', y=ticker + '_' + 'RM', label=ticker)


def draw_plot_axis(ticker, ax):
    rdf.plot(x=ticker + '_' + 'RS', y=ticker + '_' + 'RM', label=ticker, ax=ax)


def draw_plot2(ticker):
    x = np.array(rdf[ticker + '_' + 'RS'])
    y = np.array(rdf[ticker + '_' + 'RM'])
    cubic_interploation_model = interp1d(x, y, kind="cubic")
    X_ = np.linspace(x.min(), x.max(), 5000)
    Y_ = cubic_interploation_model(X_)
    return plot.plot(X_, Y_, label=ticker)


def draw_plot_axis2(ticker, ax):
    x = np.array(rdf[ticker + '_' + 'RS'])
    y = np.array(rdf[ticker + '_' + 'RM'])
    cubic_interploation_model = interp1d(x, y, kind="cubic")
    X_ = np.linspace(x.min(), x.max(), 5000)
    Y_ = cubic_interploation_model(X_)
    plot.plot(X_, Y_, label=ticker, ax=ax)


def draw_chart():
    plot.title("Sector Rotation Chart")
    plot.xlabel("Relative Strength")
    plot.ylabel("Relative Momentum")
    plot.legend(loc='upper left')
    plot.show()


def draw_plot3(ticker):
    fig, (ax1, ax2) = plot.subplots(nrows=2, sharex=True, subplot_kw=dict(frameon=True))  # frameon=False removes frames
    ax1.grid()
    ax2.grid()
    x = np.array(rdf['DATE'])
    y1 = np.array(rdf[ticker + '_' + 'RS'])
    y2 = np.array(rdf[ticker + '_' + 'RM'])
    y3 = np.array(rdf[ticker + '_' + 'PRICE'])
    ax1.plot(x, y3, color='b', label='Price', linestyle='--')
    ax1.legend(loc='upper left')
    ax2.plot(x, y1, label='Relative Strength')
    ax2.plot(x, y2, label='Relative Momentum')
    plot.xlabel("Date")
    plot.legend(loc='upper left')
    plot.title(ticker)
    plot.show()


def draw_main_chart():
    plot.figure(figsize=(12,8))
    sns.scatterplot(data=cdf, x='RS', y='RM')
    plot.title("Sector Rotation - Relative Rotation Graphs for Date: " + cdf.DATE.tail(1).item().strftime("%m/%d/%Y"))
    plot.xlabel("Relative Strength - RS Ratio")
    plot.ylabel("Relative Momentum - RM Ratio")

    plot.xlim(right=105) #xmax is your value
    plot.xlim(left=95) #xmin is your value
    plot.ylim(top=105) #ymax is your value
    plot.ylim(bottom=95) #ymin is your value

    #Country names
    for i in range(cdf.shape[0]):
        plot.text(cdf.RS[i], y=cdf.RM[i], s=cdf.TICKER[i], alpha=0.8)

    #Quadrant Marker
    plot.text(x=104, y=95, s="Weakening",alpha=0.7,fontsize=12, color='y',fontweight='bold')
    plot.text(x=95, y=95, s="Lagging",alpha=0.7,fontsize=12, color='r',fontweight='bold')
    plot.text(x=95, y=104.5, s="Improving", alpha=0.7,fontsize=12, color='b',fontweight='bold')
    plot.text(x=104, y=104.5, s="Leading", alpha=0.7,fontsize=12, color='g',fontweight='bold')

    #Mean values
    plot.axhline(y=100, color='k', linestyle='-', linewidth=1)
    plot.axvline(x=100, color='k',linestyle='-', linewidth=1)

    plot.savefig('sector_rotation_chart_' + cdf.DATE.tail(1).item().strftime("%m_%d_%Y")+'.png')
    #plot.show()


ticker_list = ['SPY', 'XLC', 'XLY', 'XLP', 'XLE', 'XLF', 'XLV', 'XLI', 'XLB', 'XLRE', 'XLK', 'XLU']
start_date = datetime(2020, 5, 1)
end_date = datetime.today()
rdf = pd.DataFrame()
count = 0

for ticker in ticker_list:
    process_tickers(ticker, start_date, end_date)

ldf = rdf.tail(1)
dates = []
tickers = []
rsv = []
rmv = []
for ticker in ticker_list:
    if ticker != 'SPY':
        tickers.append(ticker)
        dates.append(ldf['DATE'].tail(1).item())
        rsv.append(ldf[ticker+'_'+'RS'].tail(1).item())
        rmv.append(ldf[ticker+'_'+'RM'].tail(1).item())
tuples = list(zip(dates,tickers,rsv,rmv))
cdf = pd.DataFrame(tuples, columns=['DATE','TICKER', 'RS', 'RM'])
print(cdf)
cdf.to_excel('sector_rotation_summary_' + cdf.DATE.tail(1).item().strftime("%m_%d_%Y")+'.xlsx', index=False)

# for ticker in ticker_list:
#     if ticker != 'SPY':
#     if count == 0:
#         mp = draw_plot(ticker)
#         count = count + 1
#     else:
#         draw_plot_axis(ticker, mp)
#     draw_plot3(ticker)
# draw_chart()
draw_main_chart()
print(rdf)
rdf.to_excel('sector_rotation_output_' + cdf.DATE.tail(1).item().strftime("%m_%d_%Y")+'.xlsx', index=False)
