# db.py - MongoDB connection module
import os
from dotenv import load_dotenv
from pymongo import MongoClient, ASCENDING
from pymongo.errors import ConnectionFailure
from datetime import datetime
from typing import Dict, Any, List
from schema import AnomalyCollection, Anomaly

# Load environment variables
load_dotenv()

# MongoDB configuration
MONGO_URI = os.getenv("MONGODB_URI", "mongodb://127.0.0.1:27017/?directConnection=true&serverSelectionTimeoutMS=2000&appName=mongosh+2.4.2")
DB_NAME = os.getenv("DB_NAME", "mydatabase")

# Connection objects
client = None
db = None

def connect_to_database():
    global client, db
    if db is not None:
        print("Using existing database connection")
        return db

    try:
        print("Connecting to MongoDB...")
        
        # Create a new client and connect to the server
        client = MongoClient(MONGO_URI)
        
        # Verify connection
        client.admin.command('ping')
        print("Successfully connected to MongoDB!")
        
        # Connect to database
        db = client[DB_NAME]
        
        # Create indexes for anomalies collection
        create_indexes()
        
        return db

    except ConnectionFailure as e:
        print("MongoDB connection failed:", e)
        close_connection()
        raise

def create_indexes():
    """Create necessary indexes for the anomalies collection"""
    if db is None:
        raise RuntimeError("No database connection")
    
    # Get the anomalies collection
    anomalies = db["anomalies"]
    
    # Create indexes based on schema
    collection_schema = AnomalyCollection()
    for index in collection_schema.indexes:
        anomalies.create_index(
            index["keys"],
            name=index["name"]
        )
    
    print("Created indexes for anomalies collection")

def get_db():
    if db is None:
        raise RuntimeError("No database connection. Call connect_to_database first.")
    return db

def close_connection():
    global client, db
    if client:
        client.close()
        client = None
        db = None
        print("MongoDB connection closed")

def store_detection(session_id: str, detection_data: Dict[str, Any]) -> bool:
    """
    Store a detection record in MongoDB.
    """
    try:
        if db is None:
            raise RuntimeError("No database connection")
            
        # Add timestamp and session_id to the detection data
        detection_data['timestamp'] = datetime.utcnow()
        detection_data['session_id'] = session_id
        
        # Store in detections collection
        result = db["anomalies"].insert_one(detection_data)
        return bool(result.inserted_id)
    except Exception as e:
        print(f"Error storing detection: {e}")
        return False

def get_session_detections(session_id: str) -> List[Dict[str, Any]]:
    """
    Retrieve all detections for a given session.
    """
    try:
        if db is None:
            raise RuntimeError("No database connection")
            
        # Query detections for the session
        detections = list(db["anomalies"].find(
            {'session_id': session_id},
            {'_id': 0}  # Exclude MongoDB _id from results
        ).sort('timestamp', ASCENDING))
        return detections
    except Exception as e:
        print(f"Error retrieving detections: {e}")
        return []

# Optional: Connect automatically when module is imported
# connect_to_database()

if __name__ == "__main__":
    # Test the connection
    try:
        db = connect_to_database()
        print("Database connection test successful")
        print("Database name:", db.name)
        close_connection()
    except Exception as e:
        print("Connection test failed:", e)