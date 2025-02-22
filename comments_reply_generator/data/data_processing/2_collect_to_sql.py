import json
from pyrogram import Client
from pyrogram.types import Message, User
import psycopg2
from psycopg2 import OperationalError
from login import *

app = Client("my_account", api_id, api_hash)
print('@'*100)

class TGtoSQL:
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

    def emoji_handler(self, content_reactions: list):
        map_reactions = {"👍":"like", "👎":"dislike", "❤":"love", "🤔":"think", "😢":"cry", 
                        "😁":"enjoy", "🤣":"laugh", "🍾":"champagne",
                        "😡":"angry", "🙏":"pray", "🔥":"impressive", "🤡":"clown", "🤮":"vomit",
                        "🖕":"hate", "💩": "poop"}
        res_reactions = {}
        for reaction in content_reactions:
            if reaction.emoji in map_reactions:
                res_reactions[map_reactions[reaction.emoji]] = reaction.count
        return res_reactions
    
    def username_handler(self, from_user: User):
        first_name = from_user.first_name if hasattr(from_user, 'first_name') else ' '
        last_name = from_user.last_name if hasattr(from_user, 'last_name') else ' '
        username = f'({from_user.username})' if hasattr(from_user, 'username') and from_user.username is not None else ' '
        parts = [part for part in [first_name, last_name, username] if part]
        result = ' '.join(parts)
        return result

    def check_post_exists(self, post: Message):
        try:
            self.cursor.execute('''
                SELECT EXISTS (
                    SELECT 1
                    FROM POSTS
                    WHERE post_id = %s
                );
            ''', (post.id,))
            exists = self.cursor.fetchone()[0]
            return exists
        except Exception as e:
            print(f"An error occurred: {e}")
            return False
    
    def check_comment_exists(self, comment: Message):
        try:
            self.cursor.execute('''
                SELECT EXISTS (
                    SELECT 1
                    FROM COMMENTS
                    WHERE comment_id = %s
                );
            ''', (comment.id,))
            exists = self.cursor.fetchone()[0]
            return exists
        except Exception as e:
            print(f"An error occurred: {e}")
            return False

    def insert_post(self, post: Message):
        """Inserts a Telegram post into the db"""

        if self.check_post_exists(post):
            return
        insert_query = '''
        INSERT INTO POSTS (post_id, created_date, post_text, post_reactions) 
        VALUES (%s, %s, %s, %s);
        '''
        text = post.text or post.caption
        text = text.replace("\n", " ")
        if hasattr(post, 'reactions') and post.reactions and hasattr(post.reactions, 'reactions') and post.reactions.reactions: 
            post_reactions = self.emoji_handler(post.reactions.reactions)
        else:
            post_reactions = {}
        try:
            post_reactions_json = json.dumps(post_reactions)
            self.cursor.execute(insert_query, (post.id, post.date, text, post_reactions_json))
            self.connection.commit()
        except Exception as error:
            print(f"Error while inserting post {post.id}: {error}")
            self.connection.rollback()
        print(f'Post {post.id} inserted successfully')
    
    def insert_comment(self, comment: Message):
        """Insert comment or reply to a db"""

        if self.check_comment_exists(comment):
            return
        
        # Comment details
        post_id = comment.reply_to_top_message_id if comment.reply_to_top_message_id is not None else comment.reply_to_message_id
        if post_id is None or comment.text is None:
            return
        reply_to_comment_id = None if comment.reply_to_top_message_id is None else comment.reply_to_message_id
        username = self.username_handler(comment.from_user)
        comment_text = comment.text.replace("\n", " ")
        if hasattr(comment, 'reactions') and comment.reactions and hasattr(comment.reactions, 'reactions') and comment.reactions.reactions: 
            comment_reactions = self.emoji_handler(comment.reactions.reactions)
        else:
            comment_reactions = {}

        insert_query = '''
        INSERT INTO Comments (post_id, reply_to_comment_id, comment_id, created_date, username, comment_text, comment_reactions)
        VALUES (%s, %s, %s, %s, %s, %s, %s);
        '''
        try:
            comment_reactions_json = json.dumps(comment_reactions)
            self.cursor.execute(insert_query, (post_id, reply_to_comment_id, comment.id, comment.date, username, comment_text, comment_reactions_json))
            self.connection.commit()
        except Exception as error:
            print(f"Error while inserting comment {comment.id} for post {post_id}: {error}")
            self.connection.rollback()
        print(f'Comment {comment.id} inserted successfully')

    def make_positions(self):
        try:
            # Step 1: Add a new column for comment_pos if it doesn't exist
            self.cursor.execute("""
                ALTER TABLE comments ADD COLUMN IF NOT EXISTS comment_pos INT;
            """)

            # Step 2: Update the table with comment positions
            self.cursor.execute("""
                WITH numbered_posts AS (
                    SELECT 
                        post_id,
                        created_date,
                        ROW_NUMBER() OVER (PARTITION BY post_id ORDER BY created_date ASC) AS comment_pos
                    FROM
                        comments
                )
                UPDATE comments
                SET comment_pos = np.comment_pos
                FROM numbered_posts np
                WHERE comments.post_id = np.post_id AND comments.created_date = np.created_date;
            """)
            self.connection.commit()
            print("Positions added and updated successfully.")
        except Exception as e:
            print(f"An error occurred: {e}")
            self.connection.rollback()

db = TGtoSQL(**db_params)
def fetch_data(chat_id: int):
    for message in app.get_chat_history(chat_id=chat_id, limit=6_000_000):
        text = message.text or message.caption
        if text is not None:
            if message.sender_chat:
                db.insert_post(message)
            elif message.from_user:
                db.insert_comment(message)
            else:
                print('-'*100)
                print('Unknown message type, no sender_chat and no from_user in object')
                print('-'*100)
    db.make_positions()
    db.close()

def main():
    with app:
        fetch_data(DISCUSSION_CHAT_ID)

if __name__ == "__main__":
    main()