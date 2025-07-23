# 🤖 T-GNN机器学习模块部署指南

这是一个新的机器学习模块，用于污水流量监测系统的行动分类预测。

## 📋 前置要求

### 1. 环境依赖
```bash
# 切换到ml分支
git checkout feature/ml-tgnn-module

# 安装ML模块依赖
cd backend/machine_learning
pip install -r requirements.txt
```

### 2. 服务依赖
确保以下服务正在运行：
- **PostgreSQL** (TimescaleDB): 数据库
- **MLflow**: http://localhost:5002 (模型管理)
- **MinIO**: http://localhost:9000 (数据缓存)

```bash
# 启动所有服务
docker-compose up -d
```

## 🚀 快速开始

### 1. 检查系统状态
```bash
cd backend/machine_learning

# 检查可用数据
python test_prediction.py --data-info

# 检查MinIO缓存状态
python manage_cache.py health

# 检查日志系统
python manage_logs.py list
```

### 2. 数据准备
确保数据库中有：
- ✅ `monitor` 表中的设备信息
- ✅ `measurement` 表中的时序数据 (depth, flow, velocity)
- ✅ `rain_gauge` 表中的雨量数据
- ✅ `weekly_quality_check` 表中的标注数据 (actions.actions字段)
- ✅ interim格式为 `Interim1`, `Interim2` 等

### 3. 模型训练
```bash
# 基础训练 (会自动缓存数据)
python train_model.py --epochs 50 --batch-size 16

# 指定数据范围训练
python train_model.py --interims Interim1 Interim2 Interim3 --epochs 100

# 交叉验证
python train_model.py --cross-validate --cv-folds 5
```

### 4. 模型测试
```bash
# 测试单个设备预测
python test_prediction.py --monitor-id FM001 --interim Interim1

# 批量测试
python test_prediction.py --monitor-ids FM001 FM002 FM003 --interim Interim1

# 查看模型信息
python test_prediction.py --model-info
```

### 5. API集成测试
模型已集成到现有API中：
```bash
# 测试API预测 (需要后端服务运行)
curl -X POST "http://localhost:8000/api/v1/ai/predict-action" \
  -H "Content-Type: application/json" \
  -d '{"monitor_ids": ["FM001"]}'
```

## 📊 7种Action分类

模型预测以下7种行动类别：

1. **no_action_continue_monitoring** - 数据正常，继续监控
2. **investigate_storm_failure** - 暴雨响应失败，派遣现场团队
3. **investigate_dryday_failure** - 干天流量异常，检查沉积物
4. **sensor_fault_or_ragging** - 设备故障或杂物堵塞
5. **partial_data_needs_review** - 部分数据需要人工审查
6. **partial_data_no_action** - 部分数据可接受，暂无需处理
7. **recommend_remove_or_relocate** - 长期无效，建议移除/重新定位

## 🛠️ 管理工具

### 缓存管理
```bash
python manage_cache.py stats        # 查看缓存统计
python manage_cache.py clear --confirm  # 清空缓存
```

### 日志管理
```bash
python manage_logs.py list          # 查看日志文件
python manage_logs.py tail training # 查看训练日志
python manage_logs.py analyze       # 分析错误模式  
```

## 📁 日志文件位置

所有日志保存在 `logs/` 目录：
- `ml_training_YYYYMMDD.log` - 训练日志
- `ml_prediction_YYYYMMDD.log` - 预测日志
- `ml_data_YYYYMMDD.log` - 数据处理日志
- `ml_cache_YYYYMMDD.log` - 缓存操作日志
- `ml_errors_YYYYMMDD.log` - 错误汇总日志

## ⚠️ 常见问题

### 1. 导入错误
```bash
# 确保安装了所有依赖
pip install -r requirements.txt
```

### 2. 数据库连接问题
检查 `.env` 文件中的数据库配置

### 3. MLflow连接问题
确保MLflow服务在5002端口运行：
```bash
docker-compose logs mlflow
```

### 4. MinIO连接问题
确保MinIO服务在9000端口运行：
```bash
docker-compose logs minio
```

### 5. 没有训练数据
确保 `weekly_quality_check.actions.actions` 字段包含有效的action标签

## 🔄 开发流程

1. **数据准备** → 确保数据库中有标注数据
2. **首次训练** → 运行训练脚本，数据会自动缓存
3. **模型调优** → 使用缓存数据快速迭代训练
4. **模型验证** → 使用测试脚本验证预测效果
5. **生产部署** → 模型自动注册到MLflow，API自动调用

## 📞 技术支持

如有问题，请查看：
1. 日志文件：`python manage_logs.py tail <module_name>`
2. 详细文档：`backend/machine_learning/README.md`
3. 使用示例：`backend/machine_learning/example_logging_usage.py`

---
**Happy Training! 🚀**