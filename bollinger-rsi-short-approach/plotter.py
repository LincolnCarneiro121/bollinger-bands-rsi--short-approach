from settings import settings
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from processor import Processor

class Plotter(Processor):
    def __init__(self, settings=settings):
        super().__init__()

    def make_graph(self, n):
        
        make_coin_data = self.make_coin_data()
        coin_data = make_coin_data[0]
        total_profit = make_coin_data[1]
        precision = make_coin_data[2]
        coin = coin_data['coin']
        short_points_x = coin_data['short_dates']
        short_points_y = coin_data['short_points']
        buy_points_x = coin_data['profit_dates']
        buy_points_y = coin_data['profit_points']
        peak_points_x = coin_data['peak_dates']
        peak_points_y = coin_data['peak_points']

        stop_loss_points = coin_data['stop_loss_points']

        fig = make_subplots(
                rows=2, cols=1,
                specs = [[{}], [{}]],
                vertical_spacing = 0.3,
                row_heights=[0.7, 0.3]
            )

        fig.add_trace(go.Candlestick(x=coin.index,
                           open=coin['open'], high=coin['high'],
                           low=coin['low'], close=coin['close'],
                           name="Candlestick")
                           ,row=1, col=1)

        fig.add_trace(go.Scatter(x=coin.index, 
                                y=coin['upperband'], line=dict(color='green', width=1), 
                                name="ATR BUY"), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=coin.index, 
                                y=coin['lowerband'], line=dict(color='red', width=1), 
                                name="ATR SELL"), row=1, col=1)

        fig.add_trace(go.Scatter(x=coin.index, 
                                 y=coin['tracer'], line=dict(color='yellow', width=1.5), 
                                 name="Tracer"), row=1, col=1)

        fig.add_trace(go.Scatter(x=coin.index, 
                                 y=coin['sma'], line=dict(color='blue', width=1.5), 
                                 name="Middle Band"), row=1, col=1)
#
        fig.add_trace(go.Scatter(x=coin.index, 
                                 y=coin['upper_bound'], line=dict(color='purple', width=1.5), 
                                 name="Upper Band"), row=1, col=1)
#
        fig.add_trace(go.Scatter(x=coin.index, 
                                y=coin['lower_bound'], line=dict(color='red', width=1.5), 
                                 name="Lower Band"), row=1, col=1)

        fig.add_trace(go.Scatter(x=coin.index, 
                                 y=coin['rsi'], line=dict(color='red', width=1.5), 
                                 name="RSI"), row=2, col=1)
        
        fig.add_trace(go.Scatter(x=short_points_x, y=short_points_y, mode='markers', marker=dict(color='red', size=8), name='Short'))
        
        fig.add_trace(go.Scatter(x=buy_points_x, y=buy_points_y, mode='markers', marker=dict(color='blue', size=8), name='Buy'))

        fig.add_trace(go.Scatter(x=short_points_x, y=stop_loss_points, mode='markers', marker=dict(color='white', size=8), name='Stop'))

        fig.add_trace(go.Scatter(x=peak_points_x, y=peak_points_y, mode='markers', marker=dict(color='yellow', size=8), name='Peak'))


        fig.update_layout(
            title=f'Dashboard - Total Profit: {total_profit}, Precision: {precision}%',
            plot_bgcolor='rgb(17,17,17)',
            paper_bgcolor ='rgb(10,10,10)')
        
        fig.update_xaxes(color='white') 
        fig.update_yaxes(color='white')
        

        fig.update_layout(height=1000, width=1600)

        return fig

