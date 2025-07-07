from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
import logging

# Database URL
DATABASE_URL = "postgresql://suratech:suratech123@34.124.218.65:5433/meal_mate"

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class MealRecord(Base):
    __tablename__ = "meal_records"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    food_name = Column(String)
    protein = Column(Float)
    carbohydrate = Column(Float)
    fat = Column(Float)
    sodium = Column(Float)
    calories = Column(Float)
    materials = Column(Text)
    details = Column(Text)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    location_name = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

# Create tables if they don't exist
try:
    Base.metadata.create_all(bind=engine)
except Exception as e:
    logging.warning(f"Tables might already exist: {str(e)}")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close() 
