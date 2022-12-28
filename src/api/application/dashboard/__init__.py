"""Instantiate a Dash app."""
import dash
from dash import dcc
from dash import html

from .data import load_and_prepare_data
from .layout import html_layout

def init_dashboard(server):
    """Create a Plotly Dash dashboard."""
    dash_app = dash.Dash(
        server=server,
        routes_pathname_prefix="/dashboard/",
        external_stylesheets=[
            "/static/dist/css/styles.css",
            "https://fonts.googleapis.com/css?family=Lato",
        ],
    )

    data = load_and_prepare_data(server)
    # Custom HTML layout
    dash_app.index_string = html_layout
    # Create Layout
    dash_app.layout = html.Div(
        children=[
            dcc.Graph(
                id="temperature-graph",
                figure={
                    "data": [
                        {
                            "x": data["timestamp"],
                            "y": data["temperature"],
                            # https://plotly.com/javascript/reference/#scatter-line
                            "line": {
                                'color': 'blue',
                                'dash': 'dashdot',
                                'width': 2
                                }
                        }
                    ],
                    # TODO: add_hline
                    "layout": {
                        "title": "<b>Temperature</b>",
                        "height": 500,
                        "padding": 150,
                    },
                },
            ),
            dcc.Graph(
                id="rel-humidity-graph",
                figure={
                    "data": [
                        {
                            "x": data["timestamp"],
                            "y": data["humidity"],               
                            # https://plotly.com/javascript/reference/#scatter-line
                            "line": {
                                'color': 'blue',
                                'dash': 'dashdot',
                                'width': 2
                                }
                        }
                    ],
                    "layout": {
                        "title": "<b>Relative Humidity</b>",
                        "height": 500,
                        "padding": 150,
                    },                    
                },
            ),
            dcc.Graph(
                id="abs-humidity-graph",
                figure={
                    "data": [
                        {
                            "x": data["timestamp"],
                            "y": data["abs_humidity"],
                            # https://plotly.com/javascript/reference/#scatter-line
                            "line": {
                                'color': 'blue',
                                'dash': 'dashdot',
                                'width': 2
                                }                            
                        }
                    ],
                    "layout": {
                        "title": "<b>Absolute Humidity</b>",
                        "height": 500,
                        "padding": 150,
                        "color": "blue"
                    },                    
                },
            )
        ],
        id="dash-container",
    )
    return dash_app.server
