import ccxt
from settings import settings
from datetime import timedelta, datetime
import pandas as pd
import numpy as np
import numba
import utils
from playsound import playsound
import warnings
from twilio.rest import Client
import os
from trader import Trader
import time
from scipy.signal import argrelextrema
import pandas_ta as ta


class Processor(Trader):
    def __init__(self, settings=settings):
        super().__init__()
        self.start_date = datetime.now() - timedelta(days=settings['previous_days'])
        self.WINDOW = settings['WINDOW']
        self.margin = settings['margin']
        self.rsi_sup = settings['rsi_sup'] 
        self.rsi_inf = settings['rsi_inf']
        self.is_simulation = settings['just_simulation']
        self.client = Client(os.environ['SID'], os.environ['TOKEN'])
        self.from_whatsapp = 'whatsapp:' + os.environ['TWILIO_NUMBER']
        self.to_whatsapp = 'whatsapp:' + os.environ['PHONE_NUMBER']
        self.capital = settings['capital']
        self.leverage = settings['leverage']


    def make_coin_data(self):
        start_date = int(datetime.timestamp(self.start_date)) * 1000
        binance = ccxt.binanceusdm()
        trading_pair = settings['ticker']
        candles = []
        filename = 'live-data-trading-v2\\trading'

        previous_trading_data = utils.load_file(filename)

        while True:
            new_candles = binance.fetch_ohlcv(trading_pair, settings['interval'], since=start_date)
            if not new_candles or len(new_candles) == 0:
                break
            candles.extend(new_candles)
            start_date = new_candles[-1][0] + 1

        dates, opens, highs, lows, closes, volume = [], [], [], [], [], []
    
        # format the data to match the charting library
        for candle in candles:
            candle = np.around(candle, 4)
            formatted_date = datetime.fromtimestamp(candle[0] / 1000.0)
            dates.append(formatted_date)
            opens.append(candle[1])
            highs.append(candle[2])
            lows.append(candle[3])
            closes.append(candle[4])
            volume.append(candle[5])

        coin = pd.DataFrame({
            'dates': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volume,
        }).set_index('dates')

        coin.to_csv('live-data-trading-v2\sample-30.csv')

        # ATR ->

        atr_multiplier = 1.2

        current_average_high_low = (coin['high']+coin['low'])/2
        coin['atr'] = ta.atr(coin['high'], coin['low'], coin['close'], length=9)
        coin.dropna(inplace=True)
        coin['basicUpperband'] = current_average_high_low + (atr_multiplier * coin['atr'])
        coin['basicLowerband'] = current_average_high_low - (atr_multiplier * coin['atr'])
        first_upperBand_value = coin['basicUpperband'].iloc[0]
        first_lowerBand_value = coin['basicLowerband'].iloc[0]
        upperBand = [first_upperBand_value]
        lowerBand = [first_lowerBand_value]

        for i in range(1, len(coin)):
            if coin['basicUpperband'].iloc[i] < upperBand[i-1] or coin['close'].iloc[i-1] > upperBand[i-1]:
                upperBand.append(coin['basicUpperband'].iloc[i])
            else:
                upperBand.append(upperBand[i-1])

            if coin['basicLowerband'].iloc[i] > lowerBand[i-1] or coin['close'].iloc[i-1] < lowerBand[i-1]:
                lowerBand.append(coin['basicLowerband'].iloc[i])
            else:
                lowerBand.append(lowerBand[i-1])

        coin['upperband'] = upperBand
        coin['lowerband'] = lowerBand
        coin.drop(['basicUpperband', 'basicLowerband',], axis=1, inplace=True)
        
        # Intiate a signals list
        signals = [0]

        # Loop through the dataframe
        for i in range(1 , len(coin)):
            if coin['close'].iloc[i] > coin['upperband'].iloc[i]:
                signals.append(1)
            elif coin['close'].iloc[i] < coin['lowerband'].iloc[i]:
                signals.append(-1)
            else:
                signals.append(signals[i-1])

        # Add the signals list as a new column in the dataframe
        coin['signals'] = signals
        coin['signals'] = coin["signals"].shift(1) #Remove look ahead bias

        # We need to shut off (np.nan) data points in the upperband where the signal is not 1
        coin.loc[coin['signals'] == 1,'upperband'] = np.nan
        # We need to shut off (np.nan) data points in the lowerband where the signal is not -1
        coin.loc[coin['signals'] == -1,'lowerband'] = np.nan

        #AVL

        coin['MFM'] = ((coin['close'] - coin['low']) - (coin['high'] - coin['close'])) / (coin['high'] - coin['low'])
        coin['MFV'] = coin['MFM'] * coin['volume']

        coin['AVL'] = coin['MFV'].cumsum()


        # Bollinger Bands
        WINDOW = self.WINDOW

        coin['sma'] = coin['close'].rolling(WINDOW).mean()
        coin['std'] = coin['close'].rolling(WINDOW).std(ddof = 0)

        coin['bound_distance'] = [4*i for i in coin['std'].fillna(0)]
        coin['upper_bound'] = [i for i in coin['sma'] + (coin['std'] * 2)]
        coin['lower_bound'] = [i for i in coin['sma'] - (coin['std'] * 2)]

        # Tracer

        coin['tracer'] = coin['open'].ewm(span=5, adjust=False).mean()

        # RSI

        coin['change'] = coin['close'].diff()
        coin['gain'] = coin.change.mask(coin.change < 0, 0.0)
        coin['loss'] = -coin.change.mask(coin.change > 0, -0.0)

        @numba.jit
        def rma(x, n):
            """Running moving average"""
            a = np.full_like(x, np.nan)
            a[n] = x[1:n+1].mean()
            for i in range(n+1, len(x)):
                a[i] = (a[i-1] * (n - 1) + x[i]) / n
            return a

        coin['avg_gain'] = rma(coin.gain.to_numpy(), 14)
        coin['avg_loss'] = rma(coin.loss.to_numpy(), 14)
        coin['rs'] = coin.avg_gain / coin.avg_loss
        coin['rsi'] = 100 - (100 / (1 + coin.rs))

        trading_points = {
            'short_entries': [],
            'profit_exits': []
        }

        short_points_x = []
        short_points_y = []
        profit_points_x = []
        profit_points_y = []
        peak_points_x = []
        peak_points_y = []

        looking_for_profit = False
        last_short_point = None

        peak_factor = 9

        interval = coin['tracer']
        peaks = argrelextrema(interval.values, np.greater, order=peak_factor)


        # Main logic
        for i in range(1, len(coin)):  # Start from 1 to ensure there's a previous candle
        
            candle_color1 =  coin['close'].iloc[i] - coin['open'].iloc[i]
            candle_color2 =  coin['close'].iloc[i-1] - coin['open'].iloc[i-1]

            condition1 = coin['high'].iloc[i-1] > coin['upper_bound'].iloc[i-1] and coin['rsi'].iloc[i-1] > settings['rsi_sup'] and candle_color2 > 0
            condition2 = coin['low'].iloc[i-1] > coin['lower_bound'].iloc[i-1] and coin['rsi'].iloc[i-1] <= settings['rsi_low'] and candle_color2 < 0

            condition3 = coin['high'].iloc[i-2] < coin['upper_bound'].iloc[i-2]

            if not looking_for_profit and condition1:
                # Short entry point detected after the previous candle close
                short_points_x.append(coin.index[i])
                short_points_y.append(coin['open'].iloc[i])  

                peak_points_x.append(coin.index[i-1])
                peak_points_y.append(coin['upper_bound'].iloc[i-1]) 

                last_short_point = {
                    'date': coin.index[i],
                    'value': coin['open'].iloc[i]
                }
                trading_points['short_entries'].append(last_short_point)
                looking_for_profit = True  # Now looking for a profit exit

            elif looking_for_profit and condition2:
                # Profit exit point
                profit_points_x.append(coin.index[i])
                profit_points_y.append(coin['close'].iloc[i])
                profit_point = {
                    'date': coin.index[i],
                    'value': coin['close'].iloc[i]
                }
                trading_points['profit_exits'].append(profit_point)
                looking_for_profit = False  

        trading_dict = {
            "short_points": short_points_y,
            "short_dates": short_points_x,
            "profit_points": profit_points_y,
            "profit_dates": profit_points_x,
            "peak_points": peak_points_y,
            "peak_dates": peak_points_x
        }

        stop_loss_points = [i*self.margin for i in trading_dict['short_points']]

        if short_points_x and profit_points_x:
            last_trading_data = {
                'last_short': str(short_points_x[-1]),
                'last_profit': str(profit_points_x[-1]),
            }

        previous_short = datetime.strptime(previous_trading_data['last_short'], '%Y-%m-%d %H:%M:%S')
        previous_profit = datetime.strptime(previous_trading_data['last_profit'], '%Y-%m-%d %H:%M:%S')

        if short_points_x and profit_points_x:
            if short_points_x[-1] > previous_short:
                playsound('live-data-trading-v2\sounds\msn-sound_1.mp3')

                utils.save_file(last_trading_data, filename)

                self.client.messages.create(to=self.to_whatsapp, from_=self.from_whatsapp, body=f'New short detected on {self.symbol}: |{short_points_x[-1]}|{short_points_y[-1]}|')


                if not self.is_simulation:
                    self.make_short_order()


            if profit_points_x[-1] > previous_profit:
                playsound('live-data-trading-v2\sounds\\nudge.mp3')

                utils.save_file(last_trading_data, filename)

                self.client.messages.create(to=self.to_whatsapp, from_=self.from_whatsapp, body=f'New profit detected on {self.symbol}: |{profit_points_x[-1]}|{profit_points_y[-1]}|')

                if not self.is_simulation:
                    self.close_order()

        coin_data = {
            'coin': coin,
            'short_dates': trading_dict['short_dates'],
            'short_points': trading_dict['short_points'],
            'profit_dates': trading_dict['profit_dates'],
            'profit_points': trading_dict['profit_points'],
            'peak_dates': trading_dict['peak_dates'],
            'peak_points': trading_dict['peak_points'],
            'stop_loss_points': stop_loss_points,
        }

        total_profit_loss = 0
        number_loss = 0
        number_gain = 0

        for k in range(len(coin_data['profit_points'])):
            operation_interval = coin.loc[coin_data['short_dates'][k]:coin_data['profit_dates'][k]].index
            operation_points = coin['high'][operation_interval]
            stop_ = False
            for j in range(len(operation_points)):
                if operation_points.iloc[j] >= coin_data['stop_loss_points'][k]:
                    loss = (self.capital + total_profit_loss)*self.leverage*(1 - self.margin) # loss
                    total_profit_loss += loss  # loss
                    #print(f'Stop date: {operation_interval[j]}, loss: {loss}')
                    stop_ = True
                    number_loss += 1
                    break
                
            if not stop_:
                profit_loss = (coin_data['short_points'][k]/coin_data['profit_points'][k] - 1) * (self.capital + total_profit_loss) * self.leverage
                total_profit_loss += profit_loss  # Profit/loss
                if profit_loss >= 0:
                    number_gain += 1
                else:
                    number_loss += 1
                #print(f'Profit/loss: {profit_loss}', coin_data['short_points'][k]/coin_data['profit_points'][k])
            #print(f'Total profit: {total_profit_loss}')
        
        try:
            precision = (1 - number_loss/(number_loss + number_gain))*100
        except ZeroDivisionError:
            precision = None
    
        return coin_data , round(total_profit_loss,2), precision



# Call method
#Processor().make_coin_data()