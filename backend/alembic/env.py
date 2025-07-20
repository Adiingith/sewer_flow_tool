import os
import sys
from logging.config import fileConfig
from urllib.parse import quote_plus

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

# add project root directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# import all models
try:
    from backend.models.base import Base 
   

    from backend.models.measurement import Measurement
    from backend.models.rain_gauge import RainGauge
    from backend.models.monitor import Monitor
    from backend.models.presiteInstallCheck import PresiteInstallCheck
    from backend.models.weatherEvent import WeatherEvent
    from backend.models.WeeklyQualityCheck import WeeklyQualityCheck
    from backend.models.ActionResponsibility import ActionResponsibility
    
    target_metadata = Base.metadata
    print("Successfully imported models")
except ImportError as e:
    print(f"Warning: Could not import models: {e}")
    target_metadata = None

def get_database_url():
    """build database connection URL - use synchronous driver and handle special characters"""
    user = os.getenv('POSTGRES_USER')
    password = os.getenv('POSTGRES_PASSWORD')
    host = os.getenv('POSTGRES_HOST', 'postgres')
    port = os.getenv('POSTGRES_PORT', '5432')
    database = os.getenv('POSTGRES_DB')
    
    if not all([user, password, database]):
        raise ValueError(
            "Missing required environment variables: "
            "POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB"
        )
    
    # URL encode username and password, handle special characters
    encoded_user = quote_plus(user)
    encoded_password = quote_plus(password)
    
    url = f"postgresql://{encoded_user}:{encoded_password}@{host}:{port}/{database}"
    print(f"Database URL: postgresql://{encoded_user}:***@{host}:{port}/{database}")
    return url

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = get_database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    database_url = get_database_url()
    configuration = config.get_section(config.config_ini_section)
    configuration["sqlalchemy.url"] = database_url
    
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, 
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True
        )

        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
