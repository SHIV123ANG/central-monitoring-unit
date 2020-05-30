import os
import sqlite3

DATABASE_PATH = os.path.join("db", "database.db")
DATABASE_TIMEOUT = 30

def create_connection():
    '''
    Returns a sqlite3 connection instance.
    '''
    conn = sqlite3.connect(DATABASE_PATH, timeout=DATABASE_TIMEOUT)
    conn.row_factory = sqlite3.Row
    return conn


def get_person_data_from_database(id):
    '''
    id: id of person
    Returns a dictionary of person data if person id found else None
    '''
    conn = create_connection()
    cur = conn.cursor()
    query = "SELECT * FROM person WHERE id = ?"
    cur.execute(query, (id, ))
    result = cur.fetchall()
    if len(result) != 1:
        return None
    out = {}
    for k in result[0].keys():
        out[k] = result[0][k]
    return out


def search_person_by_name(name, limit=100):
    '''
    name: name of person to be searched for.
    limit: maximum number of search results to be returned.
    Returns a list of person (id, star) whose name matches the database.
    '''
    name = name.lower()
    conn = create_connection()
    cur = conn.cursor()
    query = "SELECT * FROM person WHERE name LIKE '%'||?||'%' LIMIT ?"
    cur.execute(query, (name, limit))
    res = []
    for row in cur.fetchall():
        res.append((row["id"], row["star"]))
    conn.close()
    return res


def auth_user(username, password):
    '''
    usernane: usernane
    passowrd: password
    Returns True if user is authorized else False
    '''
    conn = create_connection()
    cur = conn.cursor()
    query = "SELECT * FROM user WHERE username = ? AND password = ?"
    cur.execute(query, (username, password))
    if len(cur.fetchall()) == 1:
        return True
    conn.close()
    return False


def add_star_to_person(id, starred_persons_limit):
    '''
    id: id of person to add star to
    starred_persons_limit: maximum number of persons that can be starred
    Returns True if star added successfully else False
    '''
    conn = create_connection()
    cur = conn.cursor()
    count_query = "SELECT * FROM person WHERE star = 1"
    cur.execute(count_query)
    if len(cur.fetchall()) >= starred_persons_limit:
        return False
    update_query = "UPDATE person SET star = 1 WHERE id = ?"
    cur.execute(update_query, (id, ))
    conn.commit()
    conn.close()
    return True


def remove_star_from_person(id):
    '''
    id: id of person to remove star from
    '''
    conn = create_connection()
    cur = conn.cursor()
    update_query = "UPDATE person SET star = 0 WHERE id = ?"
    cur.execute(update_query, (id, ))
    conn.commit()
    conn.close()


def get_starred_persons():
    '''
    Returns a list of (id, star) of starred persons
    '''
    conn = create_connection()
    cur = conn.cursor()
    query = "SELECT * FROM person WHERE star = 1 ORDER BY name"
    cur.execute(query)
    res = []
    for row in cur.fetchall():
        res.append((row["id"], row["star"]))
    conn.close()
    return res
