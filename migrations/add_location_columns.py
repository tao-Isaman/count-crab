from sqlalchemy import create_engine, text
import os

# Database URL
DATABASE_URL = "postgresql://suratech:suratech123@34.124.218.65:5433/meal_mate"

def run_migration():
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
                print(f"Successfully executed: {command}")
            except Exception as e:
                print(f"Error executing {command}: {str(e)}")
        
        # Commit the changes
        connection.commit()

if __name__ == "__main__":
    run_migration() 
