"""Flask configuration."""
import os
from src.utils.helpers import get_repo_root

if os.environ.get("DEPLOY_ENV") == "production":
    ENV = 'production'
    FLASK_ENV = 'production'
    DEBUG = False
    Testing = False
    DATA_PATH = os.path.join("/home", "site", "outcome_remote.txt")
else:
    ENV = 'development'
    FLASK_ENV = 'development'
    DEBUG = True
    Testing = True
    DATA_PATH = os.path.join(get_repo_root(), "src", "example_data", "outcome_local.txt")

SECRET_KEY        = os.getenv("SECRET_KEY")
POSTGRES_HOST     = os.environ.get("POSTGRES_HOST")
POSTGRES_DBNAME   = os.environ.get("POSTGRES_DBNAME")
POSTGRES_USER     = os.environ.get("POSTGRES_USER")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD")
POSTGRES_TABLE    = os.environ.get("POSTGRES_TABLE")
