# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a sewer flow monitoring tool consisting of a FastAPI backend, React frontend, and PostgreSQL database with TimescaleDB extensions. The system manages sewer flow data, quality checks, weather events, and device monitoring with AI-powered predictions.

## Architecture

**Backend (FastAPI)**:
- `backend/main.py`: Main FastAPI application with CORS middleware
- `backend/api/v1/`: API endpoints for different functionalities
  - `monitorApi.py`: Device monitoring endpoints
  - `checkApi.py`: Quality check operations
  - `aiPredictApi.py`: AI prediction services
  - `responsibilityApi.py`: Action responsibility management
  - `functionApi.py`: Utility functions
  - `monitorAnalysisApi.py`: Monitor analysis operations
- `backend/models/`: SQLAlchemy models for database entities
- `backend/schemas/`: Pydantic schemas and data mappings
- `backend/services/`: Business logic and data processing services
- `backend/core/config.py`: Configuration management using python-decouple

**Frontend (React)**:
- React 19 with React Router for navigation
- Tailwind CSS for styling with @headlessui/react components
- Chart.js and Recharts for data visualization
- Main pages: HomePage, FleetOverview, DeviceDetail, SettingsPage
- Reusable components for device tables, KPI cards, and modals

**Database**: 
- TimescaleDB (PostgreSQL extension) for time-series data
- Alembic for database migrations
- Models include monitors, measurements, weather events, quality checks

**Additional Services**:
- Redis for caching
- MinIO for object storage
- MLflow for ML model management

## Development Commands

**Docker Development (Recommended)**:
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f [service_name]

# Rebuild services
docker-compose build

# Stop all services
docker-compose down
```

**Frontend Development**:
```bash
cd frontend
npm install
npm start          # Development server
npm run build      # Production build
npm test          # Run tests
```

**Backend Development**:
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Database migrations
alembic upgrade head
alembic revision --autogenerate -m "description"
```

**Testing**:
- Frontend: `npm test` (Jest with React Testing Library)
- Backend: No test framework configured yet

## Environment Configuration

The application requires `.env` files at multiple levels:
- Root `.env`: Database, MinIO, and shared configuration
- `frontend/.env`: React environment variables
- `backend/.env`: Backend-specific variables

Key environment variables:
- Database: `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, `POSTGRES_HOST`, `POSTGRES_PORT`
- MinIO: `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`
- Security: `SECRET_KEY`
- Storage: `UPLOAD_DIR`

## Key Features

- **Device Monitoring**: Real-time monitoring of sewer flow devices
- **Quality Checks**: Weekly quality assessments and presite installation checks
- **Weather Integration**: Weather event tracking and correlation
- **AI Predictions**: ML-powered flow predictions and anomaly detection
- **Data Visualization**: Interactive charts and dashboards
- **Action Management**: Responsibility tracking and status updates

## Data Processing

The system includes specialized services for:
- Signal rule processing (`signal_rule_engine.py`)
- Storm event detection (`storm_event_selector.py`) 
- Time series analysis (`time_series_processor.py`)
- Data quality processing (`data_processor.py`)

## Ports

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- Database: localhost:5432
- Redis: localhost:6379
- MinIO: localhost:9000 (API), localhost:9001 (Console)
- MLflow: http://localhost:5002