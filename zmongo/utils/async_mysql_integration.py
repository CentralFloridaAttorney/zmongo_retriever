# async_mysql_integration.py
import json

import mysql.connector
from mysql.connector import Error

class AsyncMySQLIntegration:
    def __init__(self, host, port, user, password, db):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.db = db
        self.connection = self.create_connection()

    def create_connection(self):
        try:
            connection = mysql.connector.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.db
            )
            if connection.is_connected():
                print("MySQL connection established")
            return connection
        except Error as e:
            print(f"Error while connecting to MySQL: {e}")
            return None

    def log_to_mysql(self, action, data):
        try:
            cursor = self.connection.cursor()
            insert_query = """
            INSERT INTO logs (action, data)
            VALUES (%s, %s)
            """
            cursor.execute(insert_query, (action, json.dumps(data)))
            self.connection.commit()
            print(f"Logged {action} action to MySQL")
        except Error as e:
            print(f"Error logging to MySQL: {e}")
        finally:
            if cursor:
                cursor.close()
