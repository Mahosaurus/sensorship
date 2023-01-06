"""Instantiate a Dash app."""
import dash
from dash import dcc
from dash import html
from dash.dependencies import Output, Input
import plotly.graph_objects as go

from src.data_visualization.graphics import PlotDashboard
from .layout import html_layout

class DashboardInit():
    def __init__(self, server):
        self.server = server
        self.dashboard_plotter = PlotDashboard(server)

    def init_dashboard(self):
        """Create a Plotly Dash dashboard."""
        dash_app = dash.Dash(
            server=self.server,
            routes_pathname_prefix="/dashboard/"
        )

        data, len_pred = self.dashboard_plotter.load_and_prepare_data()

        # Custom HTML layout
        dash_app.index_string = html_layout

        # Create Layout
        dash_app.layout = html.Div(
            children=[
                dcc.Interval(
                    id='interval-component',
                    interval=3*1000*60*10, # in milliseconds => every ten minutes
                    n_intervals=0
                ),
                dcc.Graph(
                    id="temperature-graph",
                    figure=self.dashboard_plotter.generate_plot(data, "temperature", len_pred),
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
                    figure=self.dashboard_plotter.generate_plot(data, "humidity", len_pred),
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
                    figure=self.dashboard_plotter.generate_plot(data, "abs_humidity", len_pred),
                    config={"scrollZoom": False}
                )
            ],
            id="dash-container",
        )
        self.init_callbacks(dash_app)
        return dash_app.server

    def init_callbacks(self, dash_app):
        """ Collection of callback functions """
        @dash_app.callback(
        [Output(component_id='temperature-graph', component_property='figure')],
        [Input(component_id='interval-component', component_property='n_intervals')]
        )
        def update_graph(self):
            plotter = PlotDashboard()
            data, len_pred = plotter.load_and_prepare_data()
            fig = plotter.generate_plot(data, "temperature", len_pred)
            return [go.Figure(data=fig)]

        @dash_app.callback(
        [Output(component_id='rel-humidity-graph', component_property='figure')],
        [Input(component_id='interval-component', component_property='n_intervals')]
        )
        def update_graph(self):
            plotter = PlotDashboard()
            data, len_pred = plotter.load_and_prepare_data()
            fig = plotter.generate_plot(data, "humidity", len_pred)
            return [go.Figure(data=fig)]

        @dash_app.callback(
        [Output(component_id='abs-humidity-graph', component_property='figure')],
        [Input(component_id='interval-component', component_property='n_intervals')]
        )
        def update_graph(self):
            plotter = PlotDashboard()
            data, len_pred = plotter.load_and_prepare_data()
            fig = plotter.generate_plot(data, "abs_humidity", len_pred)
            return [go.Figure(data=fig)]            
