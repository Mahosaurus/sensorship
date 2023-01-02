"""Instantiate a Dash app."""
import dash
from dash import dcc
from dash import html

from .data import load_and_prepare_data, generate_plot
from .layout import html_layout


def init_dashboard(server):
    """Create a Plotly Dash dashboard."""
    dash_app = dash.Dash(
        server=server,
        routes_pathname_prefix="/dashboard/"
    )
    data, len_pred = load_and_prepare_data(server)

    # Custom HTML layout
    dash_app.index_string = html_layout

    # Create Layout
    dash_app.layout = html.Div(
        children=[
            dcc.Interval(
                id='interval-component',
                interval=1*1000*60, # in milliseconds
                n_intervals=0
            ),
            dcc.Graph(
                id="temperature-graph",
                figure=generate_plot(data, "temperature", len_pred),
                config={"scrollZoom": False}
            ),
            dcc.Slider(
                id='temp-slider',
                min=data['slider'].min(),
                max=data['slider'].max(),
                value=data['slider'].min(),
                marks={str(year): str(year) for year in data['slider'].unique()},
                step=None
            ),
            dcc.Graph(
                id="rel-humidity-graph",
                figure=generate_plot(data, "humidity", len_pred),
                config={"scrollZoom": False}
            ),
            dcc.Slider(
                id='humidity-slider',
                min=data['slider'].min(),
                max=data['slider'].max(),
                value=data['slider'].min(),
                marks={str(year): str(year) for year in data['slider'].unique()},
                step=None
            ),
            dcc.Graph(
                id="abs-humidity-graph",
                figure=generate_plot(data, "abs_humidity", len_pred),
                config={"scrollZoom": False}
            )
        ],
        id="dash-container",
    )
    return dash_app.server
