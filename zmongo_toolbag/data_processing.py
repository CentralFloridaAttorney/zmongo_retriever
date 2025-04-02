"""
Data Processing Module
=======================

This module provides a suite of utility functions for cleaning and converting data,
flattening nested structures, extracting nested values with dot-separated keys,
and formatting text as HTML.

It is designed to handle MongoDB documents (including ObjectId and datetime),
pandas DataFrames, numpy arrays, and other complex objects. All functions are
implemented as static methods under the DataProcessing class.

If you need to control any behavior via environment variables, please add the
appropriate keys to your .env file and update the code accordingly.
"""
import ast
import html
import json
import logging
import re
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from bson import ObjectId
from bs4 import BeautifulSoup
from bson.errors import InvalidId
from dotenv import load_dotenv

# Load environment variables (if any)
load_dotenv()

# Configure module-level logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class DataProcessing:
    @staticmethod
    def clean_output_text(text: str) -> str:
        """
        Clean the provided text by removing markdown code fences (such as "```html" at the start
        and "```" at the end) and trimming extra whitespace.

        Args:
            text (str): The input text (possibly wrapped in markdown fences).

        Returns:
            str: The cleaned text.
        """
        if not isinstance(text, str):
            raise ValueError("Input text must be a string.")
        cleaned_text = text.strip()
        if cleaned_text.startswith("```html"):
            cleaned_text = cleaned_text[len("```html"):].strip()
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-len("```")].strip()
        return cleaned_text

    @staticmethod
    def convert_object_to_json(data: Any) -> Any:
        """
        Recursively convert a Python object (which may include MongoDB ObjectId, datetime,
        pandas DataFrame, numpy arrays, etc.) into a JSON-compatible structure.

        Args:
            data (Any): The data to be converted.

        Returns:
            Any: The data converted into JSON-compatible types.
        """
        def convert(obj: Any, seen: set) -> Any:
            obj_id = id(obj)
            if obj_id in seen:
                return {"__circular_reference__": obj.__class__.__name__}
            # Basic immutable types are returned as-is.
            if isinstance(obj, (int, float, bool, str, type(None))):
                return obj
            seen.add(obj_id)
            if isinstance(obj, (list, tuple, deque, set)):
                return [convert(item, seen) for item in obj]
            if isinstance(obj, dict):
                return {str(key): convert(value, seen) for key, value in obj.items()}
            if isinstance(obj, ObjectId):
                return str(obj)
            if isinstance(obj, datetime):
                return obj.isoformat()
            if isinstance(obj, pd.DataFrame):
                return obj.to_dict(orient="records")
            if isinstance(obj, pd.Series):
                return obj.to_dict()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, bytes):
                return obj.decode("utf-8", errors="replace")
            if hasattr(obj, "to_dict") and callable(obj.to_dict):
                try:
                    return convert(obj.to_dict(), seen)
                except Exception as e:
                    logger.warning(f"Error converting via to_dict: {e}")
                    return str(obj)
            if hasattr(obj, "__dict__"):
                return {attr: convert(getattr(obj, attr), seen)
                        for attr in dir(obj)
                        if not attr.startswith("_") and not callable(getattr(obj, attr))}
            return str(obj)

        try:
            return convert(data, seen=set())
        except RecursionError:
            logger.error("Maximum recursion depth exceeded while converting object to JSON.")
            return {"__error__": "Maximum recursion depth exceeded"}

    @staticmethod
    def get_oid_value(value):
        """
        Attempts to extract and return a valid ObjectId from various formats.

        Args:
            value (str | ObjectId | dict): Input to convert.

        Returns:
            ObjectId or None
        """
        if isinstance(value, ObjectId):
            return value

        if isinstance(value, dict) and '$oid' in value:
            value = value['$oid']

        if isinstance(value, str):
            try:
                if value.startswith("{'$oid'"):
                    value_dict = ast.literal_eval(value)
                    return value_dict['$oid']
                return value
            except (InvalidId, TypeError, ValueError, SyntaxError):
                return None

        return None


    @staticmethod
    def get_value(json_data: Dict[str, Any], key: str) -> Any:
        """
        Retrieve a value from a nested dictionary using a dot-separated key string.
        The function will traverse dictionaries and numeric-indexed lists.

        Args:
            json_data (Dict[str, Any]): The nested dictionary.
            key (str): The dot-separated key (e.g. "embeddings.name").

        Returns:
            Any: The value at the key path, or None if any key is not found.
        """
        keys = key.split(".")
        value = json_data
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            elif isinstance(value, list) and k.isdigit():
                index = int(k)
                value = value[index] if 0 <= index < len(value) else None
            else:
                return None
            if value is None:
                return None
        return value

    @staticmethod
    def flatten_json(json_obj: Any, prefix: str = "") -> Dict[str, Any]:
        """
        Flatten a nested JSON-like object (dict or list) into a single-level dictionary
        with dot-separated keys. The original data types of the leaf values are preserved.

        Args:
            json_obj (Any): The nested JSON-like object.
            prefix (str, optional): A prefix for keys during recursion.

        Returns:
            Dict[str, Any]: The flattened dictionary.
        """
        flat_dict: Dict[str, Any] = {}
        if isinstance(json_obj, dict):
            for key, value in json_obj.items():
                full_key = f"{prefix}.{key}" if prefix else key
                flat_dict.update(DataProcessing.flatten_json(value, full_key))
        elif isinstance(json_obj, list):
            for idx, item in enumerate(json_obj):
                full_key = f"{prefix}.{idx}" if prefix else str(idx)
                flat_dict.update(DataProcessing.flatten_json(item, full_key))
        else:
            flat_dict[prefix] = json_obj
        return flat_dict

    @staticmethod
    def get_keys_from_json(json_object: Dict[str, Any]) -> List[str]:
        """
        Retrieve all flattened keys from a nested JSON object using the flatten_json function.

        Args:
            json_object (Dict[str, Any]): The JSON object.

        Returns:
            List[str]: A list of dot-separated keys.
        """
        flat = DataProcessing.flatten_json(json_object)
        return list(flat.keys())

    @staticmethod
    def convert_text_to_html(input_data: Union[str, Dict[str, Any]]) -> str:
        """
        Convert the provided text (or a dictionary containing an 'output_text' key) into a
        formatted HTML string. This function unescapes HTML entities and returns pretty-printed HTML.

        Args:
            input_data (Union[str, Dict[str, Any]]): Either a raw string or a dictionary
                that should contain an 'output_text' key.

        Returns:
            str: A pretty-printed HTML string.

        Raises:
            ValueError: If the input_data is not a string or a dictionary containing a valid string.
        """
        logger.info(f"convert_text_to_html input_data: {input_data}")

        if isinstance(input_data, str):
            data_value = input_data
        elif isinstance(input_data, dict):
            # First convert the object to JSON-compatible format
            converted = DataProcessing.convert_object_to_json(input_data)
            data_value = converted.get("output_text")
            if not isinstance(data_value, str):
                raise ValueError("Dictionary input must contain an 'output_text' key with a string value.")
        else:
            raise ValueError("Input to convert_text_to_html must be either a string or a dictionary.")

        logger.info(f"convert_text_to_html data_value: {data_value}")
        # Parse HTML and unescape HTML entities
        soup = BeautifulSoup(data_value, "html.parser")
        for text_node in soup.find_all(string=True):
            text_node.replace_with(html.unescape(text_node))
        pretty_html = soup.prettify()
        return pretty_html

    @staticmethod
    def detect_unicode_surrogates(text: str) -> bool:
        """
        Check whether the provided text contains any Unicode surrogate pairs.

        Args:
            text (str): The text to check.

        Returns:
            bool: True if surrogate pairs are detected; False otherwise.
        """
        if not isinstance(text, str):
            raise ValueError("Input must be a string.")
        surrogate_pattern = re.compile(r"[\uD800-\uDBFF][\uDC00-\uDFFF]")
        return bool(surrogate_pattern.search(text))

    @staticmethod
    def convert_json_to_metadata(json_object: Union[Dict[str, Any], List[Any]],
                                 existing_metadata: Dict[str, Any] = None,
                                 metadata_prefix: str = "") -> Dict[str, Any]:
        """
        Convert a JSON object into a flattened metadata dictionary with dot-separated keys.

        Args:
            json_object (Union[Dict[str, Any], List[Any]]): The JSON object to flatten.
            existing_metadata (Dict[str, Any], optional): Existing metadata to update.
            metadata_prefix (str, optional): A prefix for keys.

        Returns:
            Dict[str, Any]: The flattened metadata dictionary.
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
        else:
            existing_metadata[metadata_prefix] = json_object
        return existing_metadata

    @staticmethod
    def convert_mongo_to_metadata(dict_data: Dict[str, Any],
                                  existing_metadata: Dict[str, Any] = None,
                                  metadata_prefix: str = "") -> Dict[str, Any]:
        """
        Convert a MongoDB document into a flattened metadata dictionary.
        The keys in the resulting dictionary are underscore-separated.

        Args:
            dict_data (Dict[str, Any]): The MongoDB document.
            existing_metadata (Dict[str, Any], optional): Existing metadata to update.
            metadata_prefix (str, optional): A prefix for keys.

        Returns:
            Dict[str, Any]: The flattened metadata dictionary.
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
                if isinstance(item, (dict, list)):
                    DataProcessing.convert_mongo_to_metadata(item, existing_metadata, item_prefix)
                else:
                    existing_metadata[item_prefix] = item
        else:
            existing_metadata[metadata_prefix] = dict_data

        return existing_metadata

    @staticmethod
    def get_opinion_from_zcase(zcase: Dict[str, Any]) -> str:
        """
        Extract the opinion text from a zcase document.
        This method expects a document structure with a 'casebody.data.opinions'
        list and returns the text from the first opinion.

        Args:
            zcase (Dict[str, Any]): The zcase document.

        Returns:
            str: The opinion text, or an error message if extraction fails.
        """
        try:
            casebody = zcase.get("casebody", {})
            data = casebody.get("data", {})
            opinions = data.get("opinions", [])
            if opinions and isinstance(opinions, list):
                first_opinion = opinions[0]
                return first_opinion.get("text", "No opinion text found.")
            else:
                return "No opinions found."
        except Exception as e:
            logger.error(f"Error extracting opinion from zcase: {e}")
            return f"Error extracting opinion: {e}"

    @staticmethod
    def get_values_as_list(df: pd.DataFrame, prefix: str = "embedding_") -> List[Any]:
        """
        Retrieve and flatten the values from DataFrame columns whose names start with the given prefix.

        Args:
            df (pd.DataFrame): The DataFrame containing embedding columns.
            prefix (str): The prefix to filter columns.

        Returns:
            List[Any]: A flattened list of values.
        """
        embedding_columns = [col for col in df.columns if col.startswith(prefix)]
        embeddings = df[embedding_columns].values.tolist()
        return [item for sublist in embeddings for item in sublist]

    @staticmethod
    def convert_object_to_serializable(obj: Any) -> Any:
        """
        Convert an arbitrary object into a JSON-serializable format.

        Args:
            obj (Any): The object to convert.

        Returns:
            Any: The converted, JSON-serializable object.
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
            return obj.to_dict(orient="list")
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, "__dict__"):
            return {attr: DataProcessing.convert_object_to_serializable(getattr(obj, attr))
                    for attr in dir(obj)
                    if not attr.startswith("_") and not callable(getattr(obj, attr))}
        else:
            try:
                json.dumps(obj)
                return obj
            except TypeError:
                return str(obj)

    @staticmethod
    def list_to_string(items: List[Any]) -> str:
        """
        Convert a list of items into a concatenated string.

        Args:
            items (List[Any]): The list of items.

        Returns:
            str: The concatenated string.
        """
        return "".join(map(str, items))


# -------------------- Testing / Example Usage --------------------
if __name__ == "__main__":
    # Example: Cleaning text
    raw_text = "```html\n   <p>Hello, world!</p>\n```"
    cleaned = DataProcessing.clean_output_text(raw_text)
    logger.debug(f"Cleaned text:\n{cleaned}")

    # Example: Converting an object (e.g. MongoDB document) to JSON-compatible format
    sample_data = {
        "_id": ObjectId("65fc1ec8dd5dfcff0bde5c5e"),
        "username": "overlordx",
        "profile": {"email": "overlordx@example.com", "phone": "123-456-7890"},
        "casebody": {"data": {"opinions": [{"text": "This is the opinion text."}]}},
        "dataframe": pd.DataFrame({"column1": [1, 2, 3], "column2": ["a", "b", "c"]}),
        "numpy_array": np.array([1, 2, 3, 4, 5])
    }
    converted = DataProcessing.convert_object_to_json(sample_data)
    logger.debug("Converted JSON Data:\n" + json.dumps(converted, indent=2))

    # Example: Extracting a nested value using a dot-separated key
    email_value = DataProcessing.get_value(converted, "profile.email")
    logger.debug(f"Extracted email using get_value: {email_value}")

    # Example: Converting text to HTML
    try:
        sample_html = DataProcessing.convert_text_to_html("This is a sample text.")
        logger.debug(f"Converted HTML:\n{sample_html}")
    except Exception as e:
        logger.error(f"Error in convert_text_to_html: {e}")

    # Example: Detecting Unicode surrogate pairs
    unicode_text = "This text has a surrogate pair: \uD83D\uDE00"
    has_surrogates = DataProcessing.detect_unicode_surrogates(unicode_text)
    logger.debug(f"Contains Unicode surrogates: {has_surrogates}")

    # Example: Flattening a JSON object and getting keys
    flat_json = DataProcessing.flatten_json(converted)
    logger.debug("Flattened JSON:\n" + json.dumps(flat_json, indent=2))
    keys = DataProcessing.get_keys_from_json(converted)
    logger.debug("Flattened keys from JSON:\n" + json.dumps(keys, indent=2))

    # Example: Extracting opinion text from a zcase document
    opinion_text = DataProcessing.get_opinion_from_zcase(sample_data)
    logger.debug(f"Opinion extracted from zcase: {opinion_text}")

    # Example: Getting embedding values from a DataFrame
    df_data = {
        "embedding_0": [0.1, 0.2, 0.3],
        "embedding_1": [0.4, 0.5, 0.6],
        "embedding_2": [0.7, 0.8, 0.9],
        "other_column": ["a", "b", "c"]
    }
    df = pd.DataFrame(df_data)
    embedding_values = DataProcessing.get_values_as_list(df)
    logger.debug(f"Embedding values as list: {embedding_values}")

    # Example: Converting an object to a JSON-serializable format
    serializable = DataProcessing.convert_object_to_serializable(sample_data)
    logger.debug("Serializable Data:\n" + json.dumps(serializable, indent=2))

    # Example: Converting a list to a string
    concatenated = DataProcessing.list_to_string([1, 2, 3, "abc"])
    logger.debug(f"List converted to string: {concatenated}")
