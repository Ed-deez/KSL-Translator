import os
import sqlite3
import bcrypt

DB_NAME = "ksl_translator.db"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_NAME = os.path.join(BASE_DIR, "ksl_translator.db")

def init_db():
    """Creates the users table if it doesn't already exist."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def register_user(email, password, role):
    """Hashes the password and saves the new user to the database."""
    # Convert password to bytes and hash it with a randomly generated salt
    pw_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(pw_bytes, salt)
    
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    try:
        # Store the hash as a UTF-8 string in the database
        cursor.execute(
            "INSERT INTO users (email, password, role) VALUES (?, ?, ?)", 
            (email, hashed_password.decode('utf-8'), role)
        )
        conn.commit()
        success = True
    except sqlite3.IntegrityError:
        # Triggers if the email already exists due to the UNIQUE constraint
        success = False
    finally:
        conn.close()
        
    return success

def verify_user(email, password):
    """Checks the database for the email and verifies the password hash."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute("SELECT password, role FROM users WHERE email = ?", (email,))
    user_record = cursor.fetchone()
    conn.close()
    
    if user_record:
        stored_hash, role = user_record
        
        # Check the provided plain-text password against the stored hash
        if bcrypt.checkpw(password.encode('utf-8'), stored_hash.encode('utf-8')):
            return role  # Login successful, return the user's role (Signer or Hearing User)
            
    return None  # Login failed (wrong email or wrong password)

# Initialize the database immediately when this file is run/imported
init_db()
def delete_user(email):
    """Deletes a user account from the database."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    try:
        cursor.execute("DELETE FROM users WHERE email = ?", (email,))
        conn.commit()
        success = True
    except sqlite3.Error:
        success = False
    finally:
        conn.close()
        
    return success