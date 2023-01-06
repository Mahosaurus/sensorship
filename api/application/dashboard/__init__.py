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
                    figure=self.dashboard_plotter.generate_plot("temperature"),
                    config={"scrollZoom": False}
                ),
                dcc.Graph(
                    id="rel-humidity-graph",
                    figure=self.dashboard_plotter.generate_plot("humidity"),
                    config={"scrollZoom": False}
                ),
                dcc.Graph(
                    id="abs-humidity-graph",
                    figure=self.dashboard_plotter.generate_plot("abs_humidity"),
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
            fig = plotter.generate_plot("temperature")
            return [go.Figure(data=fig)]

        @dash_app.callback(
        [Output(component_id='rel-humidity-graph', component_property='figure')],
        [Input(component_id='interval-component', component_property='n_intervals')]
        )
        def update_graph(self):
            plotter = PlotDashboard()
            fig = plotter.generate_plot("humidity")
            return [go.Figure(data=fig)]

        @dash_app.callback(
        [Output(component_id='abs-humidity-graph', component_property='figure')],
        [Input(component_id='interval-component', component_property='n_intervals')]
        )
        def update_graph(self):
            plotter = PlotDashboard()
            fig = plotter.generate_plot("abs_humidity",)
            return [go.Figure(data=fig)]
