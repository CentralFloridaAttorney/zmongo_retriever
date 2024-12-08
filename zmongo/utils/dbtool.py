import argparse
import csv
import html
import json
from urllib.parse import quote, unquote

import mysql.connector
import pandas as pd
from mysql.connector import Error
from pymongo import MongoClient
import zconstants

HTML_ESCAPE_TABLE = {
    '"': "&quot;",
    "'": "&apos;",
    ">": "&gt;",
    "<": "&lt;",
    " ": "&nbsp",
    "\n": "&#013",
    "True": "_1",
    "False": "_0"
}
HTML_UNESCAPE_TABLE = {
    "&#x27;": "'",
    "&quot;": "\""
}


class DBTool:
    def __init__(self, database_name=zconstants.DEFAULT_DATABASE, table_name=zconstants.DEFAULT_TABLE):
        self.user = zconstants.DB_USER
        self.passwd = zconstants.DB_PASSWORD
        self.host = zconstants.DB_HOST
        self.port = zconstants.DB_PORT
        self.database_name = database_name
        self.table_name = table_name
        self.db_type = zconstants.DB_TYPE  # 'mysql' or 'mongodb'

        if self.db_type == 'mysql':
            self.connection = self.get_mysql_connection()
        elif self.db_type == 'mongodb':
            self.connection = self.get_mongodb_connection()
        else:
            raise ValueError(f"Unsupported DB type: {self.db_type}")

    def get_mysql_connection(self):
        try:
            connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                passwd=self.passwd,
                port=self.port,
                database=self.database_name
            )
            return connection
        except Error as err:
            if err.errno == 1049:  # database does not exist
                connection = mysql.connector.connect(
                    host=self.host,
                    user=self.user,
                    passwd=self.passwd,
                    port=self.port,
                    database='sys'
                )
                try:
                    cursor = connection.cursor(buffered=True)
                    mysql_create_database = f"CREATE DATABASE {self.database_name};"
                    cursor.execute(mysql_create_database)
                    cursor.close()
                    connection.commit()
                    return self.get_mysql_connection()
                except Error as err:
                    print(f"Error creating database: '{err}'")
            else:
                print(f"Error: '{err}'")
            return None

    def get_mongodb_connection(self):
        try:
            client = MongoClient(self.host, self.port, username=self.user, password=self.passwd)
            return client[self.database_name]
        except Exception as e:
            print(f"Error connecting to MongoDB: {e}")
            return None

    def execute_mysql(self, query, values=None):
        try:
            cursor = self.connection.cursor(buffered=True)
            if values:
                cursor.execute(query, values)
            else:
                cursor.execute(query)
            self.connection.commit()
            if query.strip().lower().startswith('select'):
                result = cursor.fetchall()
                cursor.close()
                return result
            cursor.close()
        except Error as err:
            print(f"MySQL Error: '{err}'\nQuery: {query}")
            return None

    def add_column(self, column_name):
        if self.db_type == 'mysql':
            query = f"ALTER TABLE {self.table_name} ADD COLUMN {self._get_clean_key_string(column_name)} VARCHAR(256);"
            self.execute_mysql(query)

    def get(self, primary_key, column_name=None):
        if self.db_type == 'mysql':
            if column_name is None:
                query = f"SELECT * FROM {self.table_name} WHERE primary_key = '{self._get_clean_key_string(primary_key)}';"
            else:
                query = f"SELECT {self._get_clean_key_string(column_name)} FROM {self.table_name} WHERE primary_key = '{self._get_clean_key_string(primary_key)}';"
            result = self.execute_mysql(query)
            return self.convert_ints_and_none_to_strings(result)
        elif self.db_type == 'mongodb':
            doc = self.connection[self.table_name].find_one({"primary_key": primary_key})
            if column_name:
                return doc.get(column_name)
            return doc

    @staticmethod
    def _get_clean_key_string(s):
        return html.escape(str(s))

    def convert_ints_and_none_to_strings(self, input_list):
        if not isinstance(input_list, list):
            return self.convert_value(input_list)
        is_nested_list = all(isinstance(item, list) for item in input_list)
        if is_nested_list:
            converted_list = [[self.convert_value(item) for item in row] for row in input_list]
        else:
            converted_list = [self.convert_value(item) for item in input_list]
        return converted_list

    @staticmethod
    def convert_value(value):
        if value is None:
            return "None"
        if isinstance(value, int):
            return str(value)
        if isinstance(value, str):
            return unquote(str(value))
        return value

    def put(self, primary_key="xyzzy_primary_key", element_key=None, value=None):
        if self.db_type == 'mysql':
            if element_key is None or value is None:
                query = f"INSERT INTO {self.table_name} (primary_key) VALUES ('{self._get_clean_key_string(primary_key)}');"
                self.execute_mysql(query)
            else:
                self.add_column(element_key)
                value_escaped = self._escape_nonascii(value)
                query = f"UPDATE {self.table_name} SET {element_key} = '{value_escaped}' WHERE primary_key = '{self._get_clean_key_string(primary_key)}';"
                self.execute_mysql(query)
        elif self.db_type == 'mongodb':
            doc = {"primary_key": primary_key}
            if element_key and value:
                doc[element_key] = value
            self.connection[self.table_name].update_one({"primary_key": primary_key}, {"$set": doc}, upsert=True)

    @staticmethod
    def _escape_nonascii(input_data):
        try:
            if isinstance(input_data, dict):
                return {key: quote(str(value), safe='') for key, value in input_data.items()}
            else:
                return quote(str(input_data), safe='')
        except Exception as e:
            print(f"Error escaping non-ascii: {e}")
            return input_data

    def to_pickle(self, file_path):
        df = self.get_dataframe()
        df.to_pickle(file_path)
        print(f'Pickle saved: {file_path}')

    def get_dataframe(self):
        if self.db_type == 'mysql':
            query = f"SELECT * FROM {self.table_name};"
            rows = self.execute_mysql(query)
            columns = self.get_columns()
            df = pd.DataFrame(rows, columns=columns)
        elif self.db_type == 'mongodb':
            rows = list(self.connection[self.table_name].find({}))
            df = pd.DataFrame(rows)
        return df

    def get_columns(self):
        if self.db_type == 'mysql':
            query = f"SHOW COLUMNS FROM {self.table_name};"
            columns = self.execute_mysql(query)
            return [column[0] for column in columns]
        elif self.db_type == 'mongodb':
            sample_doc = self.connection[self.table_name].find_one({})
            return list(sample_doc.keys())

    def json2dbtool(self, primary_key_name="primary_key", json_file_path=zconstants.DEFAULT_JSON_PATH):
        with open(json_file_path, 'r') as file:
            data = json.load(file)
        primary_key = data[primary_key_name]
        self.put(primary_key=primary_key)
        for element_key, value in data.items():
            if element_key != primary_key_name:
                self.put(primary_key=primary_key, element_key=element_key, value=value)
        return self.get_dataframe()

    def csv2dbtool(self, csv_path=zconstants.DEFAULT_CSV_PATH):
        df = pd.read_csv(csv_path)
        self.add_dataframe(df, 0)
        return self.get_dataframe()

    def add_dataframe(self, df, primary_key_column_num=0):
        for index, row in df.iterrows():
            primary_key = row.iloc[primary_key_column_num]
            for column_name in df.columns:
                value = row[column_name]
                self.put(primary_key, column_name, str(value))
        print("DataFrame added to the database.")

    def delete_table(self, table_name=zconstants.DEFAULT_TABLE):
        if self.db_type == 'mysql':
            query = f"DROP TABLE {table_name};"
            self.execute_mysql(query)
        elif self.db_type == 'mongodb':
            self.connection[table_name].drop()


def main():
    parser = argparse.ArgumentParser(description="Command-line interface for DBTool")
    parser.add_argument("--get", metavar="PRIMARY_KEY", help="Retrieve data using primary key")
    parser.add_argument("--column", metavar="COLUMN_NAME", help="Optional, specify a column for --get")
    parser.add_argument("--put", metavar="PRIMARY_KEY", help="Insert or update data using primary key")
    parser.add_argument("--element", metavar="ELEMENT_KEY", help="Key for the value to be updated with --put")
    parser.add_argument("--value", metavar="VALUE", help="Value to be updated with --put")
    parser.add_argument("--json2db", metavar="JSON_FILE_PATH", help="Path to JSON file for database insertion")
    parser.add_argument("--primary_key_name", metavar="PRIMARY_KEY_NAME", help="Primary key name for JSON to DB",
                        default="primary_key")

    args = parser.parse_args()

    if not any(vars(args).values()):
        parser.print_help()
        exit()

    dbtool = DBTool()

    if args.get:
        result = dbtool.get(args.get, args.column)
        print(result)

    if args.put and args.element and args.value:
        row_num = dbtool.put(args.put, args.element, args.value)
        print(f"Updated row number: {row_num}")

    if args.json2db:
        row_id = dbtool.json2dbtool(primary_key_name=args.primary_key_name, json_file_path=args.json2db)
        print(f"Updated row with ID: {row_id}")


if __name__ == '__main__':
    main()
