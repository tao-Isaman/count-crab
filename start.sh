#!/bin/bash

# Run database migrations
echo "Running database migrations..."
python migrations/init_db.py

# Start the main application
echo "Starting the application..."
uvicorn main:app --host 0.0.0.0 --port 8080 