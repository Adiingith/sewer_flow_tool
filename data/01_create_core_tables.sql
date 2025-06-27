\echo '==== 01_create_core_tables.sql 正在执行 ===='
-- CREATE EXTENSION IF NOT EXISTS timescaledb;

--  显式声明 interval 类型，创建 hypertable
-- SELECT create_hypertable('measurement', 'time');