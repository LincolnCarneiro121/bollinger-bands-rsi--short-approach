from settings import settings
from binance.client import Client as Bclient
from binance.enums import *
from binance.exceptions import BinanceAPIException
import os
import time
import utils
from playsound import playsound

class Trader():
    def __init__(self, settings: dict = settings):
        self.binance_client = Bclient(os.environ['API_KEY'], os.environ['API_SECRET'])
        self.symbol = settings['ticker']
        self.leverage = settings['leverage']
        self.margin = settings['margin']
        self.percentage = settings['percentage']


    def create_operation_data(self):
        
        # Checks if a position is open
        positions = self.binance_client.futures_position_information()
        position_exists = any(p['symbol'] == self.symbol.replace('/', '') and float(p['positionAmt']) < 0 for p in positions)

        # Get the futures wallet balance 
        balance_info = self.binance_client.futures_account_balance()
        usdt_balance = next(item for item in balance_info if item['asset'] == 'USDT')

        capital = float(usdt_balance['balance'])*self.percentage
        
        # Leveraged value
        total_value = capital * self.leverage

        # Current market price
        current_price = float(self.binance_client.get_symbol_ticker(symbol=self.symbol)["price"])

        # Set leverage
        self.binance_client.futures_change_leverage(symbol = self.symbol, leverage=self.leverage)

        print(f"Preço de mercado atual: {current_price}")
        print(f"Alavancagem ajustada para: {self.leverage}x")
        print(f"capital sem alavancagem aplicado: {capital}")

        # Get coin info
        info = self.binance_client.futures_exchange_info()
        symbol_filters = next(item for item in info['symbols'] if item['symbol'] == self.symbol)
        
        # Get coin precision
        precision = symbol_filters['quantityPrecision']

        # Make quantity
        quantity = round(total_value / current_price, precision)
        
        # make stop loss
        stop_loss_percentage = self.margin
        stop_loss_price = current_price * stop_loss_percentage
        stop_loss_price = round(stop_loss_price, 4)

        print(f"stop price: {stop_loss_price}")

        return {
            'stop_loss_price': stop_loss_price,
            'quantity': quantity,
            'position_exists': position_exists
        }

    def make_short_order(self):

        attempt = 0
        delay = 1
        retry_count = 1

        while attempt < retry_count:
        
            try:
                data = self.create_operation_data()

                order = self.binance_client.futures_create_order(
                    symbol=self.symbol,
                    side=SIDE_SELL,
                    type=ORDER_TYPE_MARKET,
                    quantity=data['quantity']
                )
#   
                print(f"Ordem de short criada: {order}")

                stop_loss_order = self.binance_client.futures_create_order(
                    symbol=self.symbol,
                    side=SIDE_BUY,
                    type=FUTURE_ORDER_TYPE_STOP_MARKET,
                    stopPrice=data['stop_loss_price'],
                    closePosition="true"
                )
#                 
                print(f"Ordem de stop loss configurada: {stop_loss_order}")

                orderID = {
                    'orderId': str(stop_loss_order['orderId'])
                }

                utils.save_file(orderID, 'live-data-trading-v2\orderId')
                playsound('live-data-trading-v2\sounds\\new-notification-7-210334.mp3')

                break
            except BinanceAPIException as e:
                    print(f"Tentativa {attempt+1} falhou: {e.message}")
                    
                    time.sleep(delay)  # Espera antes de tentar novamente
                    attempt += 1
            except Exception as e:
                print(f"Erro não esperado: {e}")
                break
    
    def close_order(self):
        
        data = self.create_operation_data()

        try:
            if not data['position_exists']:
                print("Não existe uma posição de short para fechar.")
                return
        # Enviar uma ordem de compra de mercado para fechar a posição
            order = self.binance_client.futures_create_order(
                symbol=self.symbol, 
                side=SIDE_BUY, 
                type=ORDER_TYPE_MARKET, 
                quantity=data['quantity'],
                reduceOnly=True,
            )

            SL_id = int(utils.load_file('live-data-trading-v2\orderId')['orderId'])

            self.binance_client.futures_cancel_order(symbol=self.symbol, orderId=SL_id)

            print(f'Fechamento de ordem e stop completo.')
            
        except BinanceAPIException as e:
            print(f"Erro ao fechar a posição de short: {e.message}")
            
        except Exception as e:
            print(f"Erro não esperado: {e}")
            raise

