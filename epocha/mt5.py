from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import MetaTrader5 as mt5
# import pytz module for working with time zone
import pytz, os
import numpy as np

from stockstats import StockDataFrame

# finRL
from finrl.meta.data_processors.processor_alpaca_2 import AlpacaProcessor2
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import REST, TimeFrame


# display data on the MetaTrader 5 package
print("MetaTrader5 package author: ",mt5.__author__)
print("MetaTrader5 package version: ",mt5.__version__)