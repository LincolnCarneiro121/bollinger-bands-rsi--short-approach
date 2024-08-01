import ccxt
from datetime import datetime
import plotly.graph_objects as go
import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html
from plotter import Plotter
from settings import settings

class Websocket(Plotter):
    def __init__(self, settings=settings):
        super().__init__()
        self.app = dash.Dash(__name__, title='Dashboard', external_stylesheets=[dbc.themes.DARKLY])
        self.app.layout = html.Div([
            dcc.Graph(id='candlestick', figure='fig'),
            dcc.Interval(
                id='interval-component',
                interval= settings["sampling"] * 1000,  # in milliseconds
                n_intervals=0
            )
        ])
        self.app.callback(Output('candlestick', 'figure'), Input('interval-component', 'n_intervals'))(self.make_graph)
        

    def run(self):
        # Run the Dash app
        self.app.run_server(debug=False)

if __name__ == "__main__":
    dashboard = Websocket()
    dashboard.run()