import psycopg2
from login import *

try:
    connection = psycopg2.connect(**db_params)
    cursor = connection.cursor()

    create_posts_query = '''
    CREATE TABLE IF NOT EXISTS POSTS (
        id SERIAL PRIMARY KEY,
        post_id INTEGER NOT NULL UNIQUE,
        created_date TIMESTAMP NOT NULL,
        post_text TEXT NOT NULL,
        post_reactions JSONB DEFAULT '{}'
    );
    '''

    create_comments_query = """
    CREATE TABLE IF NOT EXISTS Comments (
        id SERIAL PRIMARY KEY,
        post_id INTEGER NOT NULL,
        reply_to_comment_id INTEGER,
        comment_id INTEGER NOT NULL UNIQUE,
        created_date TIMESTAMP NOT NULL,
        username TEXT,
        comment_text TEXT NOT NULL,
        comment_reactions JSONB
    );
    """

    cursor.execute(create_posts_query)
    cursor.execute(create_comments_query)
    connection.commit()

    print("Table(s) created successfully")

except Exception as error:
    print(f"Error while creating table: {error}")

finally:
    if cursor:
        cursor.close()
    if connection:  
        connection.close()