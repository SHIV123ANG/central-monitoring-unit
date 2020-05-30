
import os
import sqlite3
import json
from tqdm import tqdm, trange

DATABASE_PATH = os.path.join("database.db")
DATABASE_TIMEOUT = 30
PERSON_PATH = os.path.join("..", "data", "person_data")

def create_connection():
    conn = sqlite3.connect(DATABASE_PATH, timeout=DATABASE_TIMEOUT)
    conn.row_factory = sqlite3.Row
    return conn


user_table_query = '''
CREATE TABLE user(
    username TEXT PRIMARY KEY, 
    password TEXT NOT NULL
)
'''

person_table_query = '''
CREATE TABLE person(
    id TEXT PRIMARY KEY, 
    name TEXT NOT NULL, 
    star INTEGER DEFAULT 0
)
'''

def create_tables():
    conn = create_connection()
    cur = conn.cursor()
    cur.execute(user_table_query)
    cur.execute(person_table_query)
    conn.commit()
    conn.close()

def populate_database():
    conn = create_connection()
    cur = conn.cursor()
    query = "INSERT INTO user (username, password) VALUES (?, ?)"
    for user, password in [("daddy", "ripazha"), ("admin", "jumpjet")]:
        cur.execute(query, (user, password))
    query = "INSERT INTO person (id, name) VALUES (?, ?)"
    persons = os.listdir(PERSON_PATH)
    for person in tqdm(persons):
        person = os.path.join(PERSON_PATH, person)
        with open(person, "r") as f:
            person_json = json.load(f)
            m_id = person_json["id"]
            name = person_json["data"]["name"]
            cur.execute(query, (m_id, name))
    conn.commit()
    conn.close()


if __name__ == "__main__":
    create_tables()
    populate_database()
