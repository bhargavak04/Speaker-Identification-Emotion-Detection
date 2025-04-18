import os
import psycopg2
from psycopg2.extras import execute_values
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database connection parameters
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_NAME = os.getenv('DB_NAME', 'speaker_recognition')
DB_USER = os.getenv('DB_USER', 'postgres')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'password')
DB_PORT = os.getenv('DB_PORT', '5432')

def get_db_connection():
    """Create and return a database connection"""
    return psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        port=DB_PORT
    )

def initialize_database():
    """Create the necessary tables if they don't exist"""
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Create speakers table
    cur.execute('''
    CREATE TABLE IF NOT EXISTS speakers (
        id SERIAL PRIMARY KEY,
        name VARCHAR(255) UNIQUE NOT NULL,
        embedding BYTEA NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create a table for logging recognition results
    cur.execute('''
    CREATE TABLE IF NOT EXISTS recognition_logs (
        id SERIAL PRIMARY KEY,
        audio_path VARCHAR(255),
        identified_speaker VARCHAR(255),
        confidence FLOAT,
        transcription TEXT,
        emotion VARCHAR(50),
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    conn.commit()
    cur.close()
    conn.close()
    
    print("Database initialized successfully")

def store_speaker_embedding(speaker_name, embedding):
    """Store a speaker embedding in the database"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Convert numpy array to binary
        embedding_binary = embedding.tobytes()
        
        # Check if speaker already exists
        cur.execute("SELECT id FROM speakers WHERE name = %s", (speaker_name,))
        result = cur.fetchone()
        
        if result:
            # Update existing speaker
            cur.execute(
                "UPDATE speakers SET embedding = %s WHERE name = %s",
                (psycopg2.Binary(embedding_binary), speaker_name)
            )
        else:
            # Insert new speaker
            cur.execute(
                "INSERT INTO speakers (name, embedding) VALUES (%s, %s)",
                (speaker_name, psycopg2.Binary(embedding_binary))
            )
        
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Error storing speaker embedding: {str(e)}")
        return False

def get_all_speaker_embeddings():
    """Retrieve all speaker embeddings from the database"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute("SELECT name, embedding FROM speakers")
        results = cur.fetchall()
        
        speaker_embeddings = {}
        for name, embedding_binary in results:
            # Convert binary back to numpy array
            embedding = np.frombuffer(embedding_binary, dtype=np.float32)
            speaker_embeddings[name] = embedding
        
        cur.close()
        conn.close()
        return speaker_embeddings
    except Exception as e:
        print(f"Error retrieving speaker embeddings: {str(e)}")
        return {}

def log_recognition_result(audio_path, speaker, confidence, transcription, emotion):
    """Log recognition results to the database"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute(
            """INSERT INTO recognition_logs 
               (audio_path, identified_speaker, confidence, transcription, emotion) 
               VALUES (%s, %s, %s, %s, %s)""",
            (audio_path, speaker, confidence, transcription, emotion)
        )
        
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Error logging recognition result: {str(e)}")
        return False

if __name__ == "__main__":
    initialize_database()