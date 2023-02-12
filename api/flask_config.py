"""Flask configuration."""
import os
from src.utils.helpers import get_repo_root

if os.environ.get("DEPLOY_ENV") == "production":
    ENV = 'production'
    FLASK_ENV = 'production'
    DEBUG = False
    Testing = False
    DATA_PATH = os.path.join("/home", "site", "outcome_remote.txt")
    SECRET_KEY = os.environ.get("SECRET_KEY")
else:
    ENV = 'development'
    FLASK_ENV = 'development'
    DEBUG = True
    Testing = True
    DATA_PATH = os.path.join(get_repo_root(), "src", "example_data", "outcome_local.txt")
    SECRET_KEY = os.getenv("SECRET_KEY")