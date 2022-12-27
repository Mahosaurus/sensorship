
"""Initialize Flask app."""
from flask import Flask
from os import path


def init_app():
    """Construct core Flask application with embedded Dash app."""
    app = Flask(__name__, instance_relative_config=False)
    app.config.from_pyfile(path.abspath(path.dirname(path.dirname(__file__))) + "/flask_config.py")
    app.config["CSV_PATH"] = path.abspath(path.dirname(path.dirname(__file__))) + "/data/311-calls.csv"

    with app.app_context():
        # Import parts of our core Flask app
        from . import routes

        # Import Dash application
        from .dashboard import init_dashboard

        app = init_dashboard(app)

        return app