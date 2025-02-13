# data_processing.py 112524_0436
import html
import json
import logging
import re
from collections import deque
from datetime import datetime
from bson import ObjectId
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np


class DataProcessing:
    @staticmethod
    def clean_output_text(text):
        # Remove the ` ```html` at the beginning and ` ``` ` at the end
        cleaned_text = text.strip()  # Remove leading/trailing whitespaces
        if cleaned_text.startswith('```html'):
            cleaned_text = cleaned_text[len('```html'):].strip()
        if cleaned_text.endswith('```'):
            cleaned_text = cleaned_text[:-len('```')].strip()

        return cleaned_text

    @staticmethod
    def convert_object_to_json(data):
        """
        Recursively converts MongoDB ObjectId instances, pandas DataFrames, numpy arrays, and other data structures
        to their string or JSON-compatible representations.

        Args:
            data (any): The data to be converted, which can be a dictionary, list, ObjectId, pandas DataFrame,
                        numpy array, or other types.

        Returns:
            any: The data with ObjectId instances, DataFrames, and arrays converted to strings or JSON-compatible formats.
        """

        def convert(obj, seen):
            obj_id = id(obj)
            if obj_id in seen:
                return {"__circular_reference__": obj.__class__.__name__}
            # For immutable basic types, no need to track
            if isinstance(obj, (int, float, bool, str, type(None))):
                return obj
            seen.add(obj_id)

            if isinstance(obj, (list, tuple, set, deque)):
                return [convert(item, seen) for item in obj]
            elif isinstance(obj, dict):
                new_dict = {}
                for key, value in obj.items():
                    # JSON requires string keys
                    new_key = str(key)
                    new_dict[new_key] = convert(value, seen)
                return new_dict
            elif isinstance(obj, ObjectId):
                return str(obj)
            elif isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict(orient='records')  # List of dicts
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (pd.Timestamp, pd.Timedelta)):
                return str(obj)
            elif isinstance(obj, bytes):
                return obj.decode('utf-8', errors='replace')
            elif isinstance(obj, (int, float)) and len(str(obj)) > 10:
                # Assuming it's a Unix timestamp if it's a large integer
                try:
                    return datetime.utcfromtimestamp(obj).isoformat() + 'Z'
                except (OverflowError, OSError, ValueError):
                    return obj
            elif hasattr(obj, 'to_dict') and callable(obj.to_dict):
                try:
                    return convert(obj.to_dict(), seen)
                except Exception as e:
                    return {"__to_dict_error__": str(e)}
            elif hasattr(obj, '__dict__'):
                return {key: convert(value, seen)
                        for key, value in obj.__dict__.items()
                        if not key.startswith('_')}
            elif hasattr(obj, '__slots__'):
                return {slot: convert(getattr(obj, slot), seen)
                        for slot in obj.__slots__
                        if hasattr(obj, slot)}
            else:
                return str(obj)

        try:
            converted_data = convert(data, seen=set())
            return converted_data
        except RecursionError:
            return {"__error__": "Maximum recursion depth exceeded"}

    @staticmethod
    def get_value(json_data, key):
        """
        Retrieves a value from a nested dictionary structure using a dot-separated key string.

        Args:
            json_data (dict): The dictionary from which to retrieve the value.
            key (str): A dot-separated string representing the key path.

        Returns:
            any: The value retrieved from the nested dictionary, or None if any key along the path is not found.
        """
        keys = key.split(".")
        value = json_data
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            elif isinstance(value, list) and k.isdigit():
                index = int(k)
                value = value[index] if 0 <= index < len(value) else None
            else:
                return None
        return value

    @staticmethod
    def convert_text_to_html(input_data):
        """
        Converts the provided text or JSON object into a formatted HTML string.

        Args:
            input_data (Union[str, dict]): The text or JSON object to be converted to HTML.

        Returns:
            str: The text wrapped in HTML tags.
        """
        logging.info(f"convert_text_to_html input_data: {input_data}")

        if isinstance(input_data, str):
            data_value = input_data
        elif isinstance(input_data, dict):
            data_object = DataProcessing.convert_object_to_json(input_data)
            data_value = data_object.get('output_text')
            if not isinstance(data_value, str):
                raise ValueError("Input to convert_text_to_html must be a string.")
        else:
            raise ValueError("Input to convert_text_to_html must be either a string or a dictionary.")

        logging.info(f"convert_text_to_html data_value: {data_value}")

        soup = BeautifulSoup(data_value, 'html.parser')
        for text_node in soup.find_all(string=True):
            text_node.replace_with(html.unescape(text_node))
        pretty_html = soup.prettify()
        return pretty_html.encode("utf-8").decode("utf-8")

    @staticmethod
    def detect_unicode_surrogates(text):
        """
        Detects if the provided text contains any Unicode surrogate pairs.

        Args:
            text (str): The text to be checked for surrogate pairs.

        Returns:
            bool: True if surrogate pairs are found, False otherwise.
        """
        surrogate_pair_pattern = re.compile(r'[\uD800-\uDBFF][\uDC00-\uDFFF]')
        return bool(surrogate_pair_pattern.search(text))

    @staticmethod
    def convert_json_to_metadata(json_object, existing_metadata=None, metadata_prefix=''):
        """
        Converts a JSON object into a flattened metadata dictionary with dot-separated keys.

        Args:
            json_object (dict or list): The JSON object to be converted.
            existing_metadata (dict, optional): An existing metadata dictionary to update.
            metadata_prefix (str, optional): A prefix for the metadata keys.

        Returns:
            dict: A metadata dictionary with flattened keys.
        """
        if existing_metadata is None:
            existing_metadata = {}

        if isinstance(json_object, dict):
            for key, value in json_object.items():
                new_prefix = f"{metadata_prefix}.{key}" if metadata_prefix else key
                DataProcessing.convert_json_to_metadata(value, existing_metadata, new_prefix)
        elif isinstance(json_object, list):
            for idx, item in enumerate(json_object):
                item_prefix = f"{metadata_prefix}.{idx}" if metadata_prefix else str(idx)
                DataProcessing.convert_json_to_metadata(item, existing_metadata, item_prefix)
        elif hasattr(json_object, 'to_dict'):
            DataProcessing.convert_json_to_metadata(json_object.to_dict(), existing_metadata, metadata_prefix)
        elif hasattr(json_object, '__dict__'):
            DataProcessing.convert_json_to_metadata(json_object.__dict__, existing_metadata, metadata_prefix)
        else:
            existing_metadata[metadata_prefix] = str(json_object)

        return existing_metadata

    @staticmethod
    def convert_mongo_to_metadata(dict_data, existing_metadata=None, metadata_prefix=''):
        """
        Converts a MongoDB document into a flattened metadata dictionary with underscore-separated keys.

        Args:
            dict_data (dict): The MongoDB document to be converted.
            existing_metadata (dict, optional): An existing metadata dictionary to update.
            metadata_prefix (str, optional): A prefix for the metadata keys.

        Returns:
            dict: A metadata dictionary with flattened keys.
        """
        if existing_metadata is None:
            existing_metadata = {}

        if dict_data is None:
            return existing_metadata

        if isinstance(dict_data, dict):
            for key, value in dict_data.items():
                new_prefix = f"{metadata_prefix}_{key}" if metadata_prefix else key
                DataProcessing.convert_mongo_to_metadata(value, existing_metadata, new_prefix)
        elif isinstance(dict_data, list):
            for idx, item in enumerate(dict_data):
                item_prefix = f"{metadata_prefix}_{idx}" if metadata_prefix else str(idx)
                if isinstance(item, dict) or isinstance(item, list):
                    item_json = DataProcessing.convert_object_to_json(item)
                    DataProcessing.convert_mongo_to_metadata(item_json, existing_metadata, item_prefix)
                else:
                    existing_metadata[item_prefix] = str(item)
        else:
            existing_metadata[metadata_prefix] = str(dict_data)

        return existing_metadata

    @staticmethod
    def get_keys_from_json(json_object):
        """
        Retrieves all the keys from a nested JSON object.

        Args:
            json_object (dict): The JSON object to extract keys from.

        Returns:
            list: A list of keys in the JSON object.
        """
        this_metadata = DataProcessing.convert_json_to_metadata(json_object=json_object)
        return list(this_metadata.keys())

    @staticmethod
    def get_opinion_from_zcase(zcase):
        """
        Extracts the opinion text from a zcase document.
        this is the technique without using the path-like-dot-separated-keys
        Args:
            zcase (dict): The zcase document containing the opinion.

        Returns:
            str: The extracted opinion text or an error message if not found.
        """
        try:
            casebody = zcase.get('casebody')
            data = casebody.get('data')
            opinions = data.get('opinions')
            this_opinion = opinions[0]
            opinion_text = this_opinion.get('text')
            return opinion_text
        except Exception as e:
            return f"No opinion: {str(e)}"


    @staticmethod
    def get_values_as_list(df: pd.DataFrame, prefix: str = 'embedding_') -> list:
        """
        Retrieves the values of all columns that start with a given prefix and returns them as a list.

        Args:
            df (pd.DataFrame): The DataFrame containing the embedding columns.
            prefix (str): The prefix used to identify the embedding columns.

        Returns:
            list: A list containing the values of the embedding columns.
        """
        # Filter columns that start with the prefix
        embedding_columns = [col for col in df.columns if col.startswith(prefix)]

        # Extract the values of these columns and return as a list
        embeddings = df[embedding_columns].values.tolist()

        # Convert list of lists to list of embeddings
        return [item for sublist in embeddings for item in sublist]

    @staticmethod
    def convert_object_to_serializable(obj):
        """
        Converts an object to a serializable format.

        Args:
            obj (any): The object to be converted.

        Returns:
            any: The serializable object.
        """
        if isinstance(obj, dict):
            return {key: DataProcessing.convert_object_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [DataProcessing.convert_object_to_serializable(item) for item in obj]
        elif isinstance(obj, ObjectId):
            return str(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='list')
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, "__dict__"):
            return {attr: DataProcessing.convert_object_to_serializable(getattr(obj, attr)) for attr in dir(obj)
                    if not attr.startswith(('__', '_')) and not callable(getattr(obj, attr))}
        else:
            try:
                return json.dumps(obj, default=str)
            except TypeError:
                return str(obj)

    @staticmethod
    def list_to_string(items: list) -> str:
        """
        Convert a list of items into a single string with items in the same order as given.

        Args:
            items (list): The list of items to be converted into a string.

        Returns:
            str: A single string with all items concatenated in order.
        """
        return ''.join(map(str, items))

# Example usage
if __name__ == "__main__":
    # Sample DataFrame with embedding columns
    data = {
        'embedding_0': [0.1, 0.2, 0.3],
        'embedding_1': [0.4, 0.5, 0.6],
        'embedding_2': [0.7, 0.8, 0.9],
        'other_column': ['a', 'b', 'c']
    }

    this_df = pd.DataFrame(data)

    # Get the embedding values as a list
    embedding_values = DataProcessing.get_values_as_list(this_df)
    print(embedding_values)


    sample_data = {
        "_id": ObjectId("65fc1ec8dd5dfcff0bde5c5e"),
        "username": "overlordx",
        "password": "pbkdf2:sha256:600000$HGQqogvlkV3uWD0f$f760a60a55754822f44935f7aa57ed0f569dc7cfd2654f6451c6f945b2212f44",
        "profile": {
            "email": "overlordx@example.com",
            "phone": "123-456-7890"
        },
        "casebody": {
            "data": {
                "opinions": [
                    {
                        "text": "This is the opinion text."
                    }
                ]
            }
        },
        "dataframe": pd.DataFrame({
            "column1": [1, 2, 3],
            "column2": ["a", "b", "c"]
        }),
        "numpy_array": np.array([1, 2, 3, 4, 5])
    }

    # Convert MongoDB document to JSON-compatible format
    json_data = DataProcessing.convert_object_to_json(sample_data)
    print("Converted JSON Data:", json.dumps(json_data, indent=2))

    # Get value using dot-separated key
    email = DataProcessing.get_value(sample_data, "profile.email")
    print("Email:", email)

    # Convert text to HTML
    html_text = DataProcessing.convert_text_to_html("This is a sample text.")
    print("HTML Text:", html_text)

    # Detect Unicode surrogates
    unicode_text = "This is a sample text with a surrogate pair: \uD83D\uDE00"
    has_surrogates = DataProcessing.detect_unicode_surrogates(unicode_text)
    print("Contains Unicode Surrogates:", has_surrogates)

    # Convert JSON to Metadata
    metadata = DataProcessing.convert_json_to_metadata(sample_data)
    print("Metadata:", json.dumps(metadata, indent=2))

    # Convert MongoDB document to Metadata
    mongo_metadata = DataProcessing.convert_mongo_to_metadata(sample_data)
    print("Mongo Metadata:", json.dumps(mongo_metadata, indent=2))

    # Get keys from JSON
    keys = DataProcessing.get_keys_from_json(sample_data)
    print("Keys from JSON:", keys)

    # Get opinion from zcase
    opinion_text = DataProcessing.get_opinion_from_zcase(sample_data)
    print("Opinion Text:", opinion_text)

    # Convert object to serializable format
    serializable_data = DataProcessing.convert_object_to_serializable(sample_data)
    print("Serializable Data:", json.dumps(serializable_data, indent=2))
