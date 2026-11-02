import sqlite3
import os

db_path = os.path.join("instance", "learning_platform.db")
conn = sqlite3.connect(db_path)
c = conn.cursor()

try:
    c.execute("ALTER TABLE assessment ADD COLUMN raw_math_score FLOAT")
    c.execute("ALTER TABLE assessment ADD COLUMN raw_science_score FLOAT")
    c.execute("ALTER TABLE assessment ADD COLUMN raw_english_score FLOAT")
    c.execute("ALTER TABLE assessment ADD COLUMN raw_social_score FLOAT")
    # For existing rows, just set them to the assessment score or a default
    c.execute("UPDATE assessment SET raw_math_score = math_score, raw_science_score = science_score, raw_english_score = english_score, raw_social_score = social_score")
    conn.commit()
    print("Columns added successfully.")
except Exception as e:
    print(f"Error: {e}")

conn.close()
