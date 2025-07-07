from sqlalchemy import create_engine, text
import os
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database URL
DATABASE_URL = "postgresql://suratech:suratech123@34.124.218.65:5433/meal_mate"

def wait_for_db(max_retries=5, retry_interval=5):
    """Wait for database to be ready"""
    for i in range(max_retries):
        try:
            engine = create_engine(DATABASE_URL)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Database is ready!")
            return True
        except Exception as e:
            if i < max_retries - 1:
                logger.warning(f"Database not ready. Retrying in {retry_interval} seconds... ({i+1}/{max_retries})")
                time.sleep(retry_interval)
            else:
                logger.error(f"Could not connect to database after {max_retries} attempts")
                raise e

def run_migrations():
    """Run all database migrations"""
    try:
        # Wait for database to be ready
        wait_for_db()
        
        # Create SQLAlchemy engine
        engine = create_engine(DATABASE_URL)
        
        # SQL commands to add new columns
        commands = [
            "ALTER TABLE meal_records ADD COLUMN IF NOT EXISTS latitude FLOAT",
            "ALTER TABLE meal_records ADD COLUMN IF NOT EXISTS longitude FLOAT",
            "ALTER TABLE meal_records ADD COLUMN IF NOT EXISTS location_name VARCHAR"
        ]
        
        # Execute each command
        with engine.connect() as connection:
            for command in commands:
                try:
                    connection.execute(text(command))
                    logger.info(f"Successfully executed: {command}")
                except Exception as e:
                    logger.error(f"Error executing {command}: {str(e)}")
                    raise e
            
            # Commit the changes
            connection.commit()
            logger.info("All migrations completed successfully!")
            
    except Exception as e:
        logger.error(f"Migration failed: {str(e)}")
        raise e

if __name__ == "__main__":
    run_migrations() 
