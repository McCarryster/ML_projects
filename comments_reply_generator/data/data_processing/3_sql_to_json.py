import psycopg2
from psycopg2 import OperationalError
import json
import os
from login import *

class SQL_to_json:
    def __init__(self, dbname, user, password, host='localhost', port='5432'):
        self.connection = None
        self.cursor = None
        self.db_params = {
            'dbname': dbname,
            'user': user,
            'password': password,
            'host': host,
            'port': port
        }
        self.connect()

    def connect(self):
        try:
            self.connection = psycopg2.connect(**self.db_params)
            self.cursor = self.connection.cursor()
        except OperationalError as e:
            print(f"Error connecting to the db: {e}")

    def close(self):
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()

    def __del__(self):
        """Ensures that the connection is closed when the object is garbage collected"""
        self.close()

    def check_post_id_exists(self, post_id: int, from_where: str):
        try:
            self.cursor.execute(f'''
                SELECT EXISTS (
                    SELECT 1
                    FROM {from_where}
                    WHERE post_id = %s
                );
            ''', (post_id,))
            exists = self.cursor.fetchone()[0]
            return exists
        except Exception as e:
            print(f"An error occurred: {e}")
            return False

    def check_comment_exists(self, comment_id: int):
        try:
            self.cursor.execute('''
                SELECT EXISTS (
                    SELECT 1
                    FROM COMMENTS
                    WHERE comment_id = %s
                );
            ''', (comment_id,))
            exists = self.cursor.fetchone()[0]
            return exists
        except Exception as e:
            print(f"An error occurred: {e}")
            return False

    def get_posts(self):
        self.cursor.execute("""SELECT 
                            post_id, post_text, post_reactions
                            FROM posts""")
        posts = self.cursor.fetchall()
        return posts

    def get_comments_for_post(self, post_id: int):
        self.cursor.execute("""
            SELECT post_id, reply_to_comment_id, comment_id, created_date, username, comment_text, comment_reactions, comment_pos
            FROM comments
            WHERE post_id = %s
        """, (post_id,))
        comments = self.cursor.fetchall()
        return comments

    def get_one_comment(self, comment_id: int):
        self.cursor.execute("""
                SELECT post_id, reply_to_comment_id, comment_id, created_date, username, comment_text, comment_reactions, comment_pos
                FROM comments
                WHERE comment_id = %s
            """, (comment_id,))
        reply_comment = self.cursor.fetchall()
        return reply_comment

db = SQL_to_json(**db_params)
data = []
posts = db.get_posts()
for post in posts:
    print(f'Processing {post[0]}...')

    if not db.check_post_id_exists(post[0], 'COMMENTS'):     # Check if post_id exists in comments table
        continue
    comments = db.get_comments_for_post(post[0])
    for comment in comments:
        if not db.check_post_id_exists(post[0], 'POSTS'):    # Check if post_id exists in posts table
            continue

        # comment_generation
        if comment[1] is None:
            data.append(
                {
                    "input": {
                        "task": "comment_generation",
                        "post": post[1],
                        "post_reactions": post[2],
                        "position": comment[7],
                        "desired_reactions": comment[6]
                    },
                    "output": comment[5]
                }
            )
        # comment_generation

        # reply_generation
        else:
            comment_data = db.get_one_comment(comment[1])
            if comment_data:
                input_up_comment_data = comment_data[0]
            else:
                continue
            curr_reply_to_comment_id = input_up_comment_data[1]
            reply_history = []
            while curr_reply_to_comment_id is not None:
                if not db.check_comment_exists(curr_reply_to_comment_id): # Check if comment_id exists in comments table
                    break
                reply = db.get_one_comment(curr_reply_to_comment_id)[0]
                if not reply:
                    break
                reply_history.append(
                    {
                        "username": reply[4],
                        "comment": reply[5],
                        "reactions": reply[6],
                    }
                )
                curr_reply_to_comment_id = reply[1]
            data.append(
                {
                    "input": {
                        "task": "reply_generation",
                        "post": post[1],
                        "post_reactions": post[2],
                        "username": input_up_comment_data[4],
                        "comment_to_reply": input_up_comment_data[5],
                        "comment_reactions": input_up_comment_data[6],
                        "history": reply_history[::-1],
                        "desired_reactions": comment[6]
                    },
                    "output": comment[5]
                }
            )
        # reply_generation
db.close()

script_dir = os.path.dirname(__file__)
output_dir = os.path.join(script_dir, '../data_types/processed_data/json_data.json')
with open(output_dir, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4)
print(f'Data stored successfully!') 