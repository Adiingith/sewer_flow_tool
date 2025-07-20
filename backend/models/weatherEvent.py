from sqlalchemy import Column, Integer, String, Text, TIMESTAMP, JSON, Identity, UniqueConstraint
from sqlalchemy.orm import relationship
from .base import Base

class WeatherEvent(Base):
    __tablename__ = 'weather_event'

    id = Column(Integer, primary_key=True, server_default=Identity())
    event_type = Column(Text, nullable=False)      # 'weekly_storm' , 'weekly_dry_day' , 'fire_storm' , 'fire_dry_day'
    start_time = Column(TIMESTAMP(timezone=True), nullable=False)
    end_time = Column(TIMESTAMP(timezone=True), nullable=False)
    area = Column(Text, nullable=False)            # area
    interim = Column(Text, nullable=False)         # interim
    storm_type = Column(Text)                      # only for storm event, A/B/C
    dry_day_number = Column(Integer)               # only for dry day event
    coverage = Column(Text)                        # only for storm event
    event_comment = Column(JSON)                   # metadata
    rain_gauge_monitor_id = Column(Integer, nullable=False)  # store monitor.id (rain gauge device ID)

    __table_args__ = (
        UniqueConstraint('rain_gauge_monitor_id', 'interim', 'event_type', 'area', 'start_time', 'end_time', name='uq_weather_event_unique'),
    )

    # no foreign key constraint, get device information through join query