import ccxt
from settings import settings
from datetime import timedelta, datetime
import pandas as pd
import numpy as np
import numba
import utils
import os
import itertools

class Optimizer():
    def __init__(self):
        self.start_date = datetime.now() - timedelta(days=30)
        self.WINDOW = None
        self.margin = None
        self.rsi_sup = None
        self.rsi_inf = None
        self.capital = 10
        self.leverage = None

    def make_coin_data(self):
        
        coin = pd.read_csv('live-data-trading-v2\sample-30.csv')

        # Bollinger Bands
        WINDOW = self.WINDOW

        coin['sma'] = coin['close'].rolling(WINDOW).mean()
        coin['std'] = coin['close'].rolling(WINDOW).std(ddof = 0)

        coin['bound_distance'] = [4*i for i in coin['std'].fillna(0)]
        coin['upper_bound'] = np.around([i for i in coin['sma'] + (coin['std'] * 2)],4)
        coin['lower_bound'] = np.around([i for i in coin['sma'] - (coin['std'] * 2)],4)

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

        looking_for_profit = False
        last_short_point = None

        for i in range(1, len(coin)):  # Start from 1 to ensure there's a previous candle
            if not looking_for_profit and coin['high'].iloc[i-1] >= coin['upper_bound'].iloc[i-1] and coin['rsi'].iloc[i-1] > self.rsi_sup:
                # Short entry point detected after the previous candle close
                short_points_x.append(coin.index[i-1])
                short_points_y.append(coin['upper_bound'].iloc[i-1])

                last_short_point = {
                    'date': coin.index[i],
                    'value': coin['upper_bound'].iloc[i]
                }
                trading_points['short_entries'].append(last_short_point)
                looking_for_profit = True  # Now looking for a profit exit
            elif looking_for_profit and coin['low'].iloc[i] <= coin['lower_bound'].iloc[i] and coin['rsi'].iloc[i] < self.rsi_inf:
                # Profit exit point
                profit_points_x.append(coin.index[i])
                profit_points_y.append(coin['lower_bound'].iloc[i])

                profit_point = {
                    'date': coin.index[i],
                    'value': coin['lower_bound'].iloc[i]
                }
                trading_points['profit_exits'].append(profit_point)
                looking_for_profit = False 

        trading_dict = {
            "short_points": short_points_y,
            "short_dates": short_points_x,
            "profit_points": profit_points_y,
            "profit_dates": profit_points_x,
        }

        stop_loss_points = [i*self.margin for i in trading_dict['short_points']]

        coin_data = {
            'coin': coin,
            'short_dates': trading_dict['short_dates'],
            'short_points': trading_dict['short_points'],
            'profit_dates': trading_dict['profit_dates'],
            'profit_points': trading_dict['profit_points'],
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
                    loss = (self.capital + total_profit_loss)*self.leverage*(1 - self.margin) # Loss
                    total_profit_loss += loss  # Loss
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
                #print(f'Profit/Loss: {profit_loss}', coin_data['short_points'][k]/coin_data['profit_points'][k])
            #print(f'Total profit: {total_profit_loss}')

        precision = (1 - number_loss/(number_loss + number_gain))*100
    
        return coin_data , round(total_profit_loss,2), precision
    

    def optimize_parameters(optimizer):
        # Definindo os intervalos para cada parâmetro
        WINDOW_range = np.arange(11,16,1)  # Por exemplo, de 5 a 20, de 5 em 5
        margin_range = [1.3]  # De 1.0 a 1.5, incrementos de 0.1
        rsi_sup_range = range(60, 78, 1)  # Por exemplo, de 65 a 80, de 5 em 5
        rsi_inf_range = range(30, 55, 1)  # De 15 a 30, de 5 em 5
        leverage_range = [5]  # De 1x a 5x

        total_iter = len(WINDOW_range) * len(margin_range) * len(rsi_sup_range) * len(rsi_inf_range) * len(leverage_range)
        print(f'TOTAL ITERATIONS NEEDED: {total_iter} <-------')

        # Melhores parâmetros e resultados iniciais
        best_precision = 0
        best_profit_loss = float('-inf')
        best_parameters = {}
        iteration = 0

        # Iterar sobre todas as combinações possíveis dos intervalos de parâmetros
        for WINDOW, margin, rsi_sup, rsi_inf, leverage in itertools.product(WINDOW_range, margin_range, rsi_sup_range, rsi_inf_range, leverage_range):
            # Atualizar os parâmetros do otimizador
            print(f'iteration: n#{iteration}')
            optimizer.WINDOW = WINDOW
            optimizer.margin = margin
            optimizer.rsi_sup = rsi_sup
            optimizer.rsi_inf = rsi_inf
            optimizer.leverage = leverage

            # Executar a simulação com os novos parâmetros
            _, profit_loss, precision = optimizer.make_coin_data()

            # Calcula uma métrica de desempenho combinada
            # Aqui usamos uma soma ponderada simples como exemplo
            combined_score = profit_loss  # Ajuste os pesos conforme necessário

            # Verificar se encontramos uma nova melhor combinação
            if combined_score > best_profit_loss:
                best_precision = precision
                best_profit_loss = profit_loss
                best_parameters = {
                    'WINDOW': WINDOW,
                    'margin': margin,
                    'rsi_sup': rsi_sup,
                    'rsi_inf': rsi_inf,
                    'leverage': leverage
                }

            iteration+=1

            if iteration == 10000:
                break

        print(f"Best parameters: {best_parameters}")
        print(f"Best precision: {best_precision}")
        print(f"Best total_profit_loss: {best_profit_loss}")

Optimizer().optimize_parameters()

            
# Best parameters: {'WINDOW': 11, 'margin': 1.3000000000000003, 'rsi_sup': 64, 'rsi_inf': 24, 'leverage': 5}
#Best precision: 81.84210526315789
#Best total_profit_loss: 1835.58


    