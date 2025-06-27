import os
import sys
from logging.config import fileConfig
from urllib.parse import quote_plus

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

# 添加项目根目录到 Python 路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# 导入所有模型
try:
    from backend.models.base import Base 
    

    import backend.models.measurement
    import backend.models.monitor
    import backend.models.presiteInstallCheck
    import backend.models.StormEvent
    import backend.models.DryDayEvent
    import backend.models.WeeklyQualityCheck
    import backend.models.ActionResponsibility
    
    target_metadata = Base.metadata
    print("Successfully imported models")
except ImportError as e:
    print(f"Warning: Could not import models: {e}")
    target_metadata = None

def get_database_url():
    """构建数据库连接URL - 使用同步驱动并处理特殊字符"""
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
    
    # URL 编码用户名和密码，处理特殊字符
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
