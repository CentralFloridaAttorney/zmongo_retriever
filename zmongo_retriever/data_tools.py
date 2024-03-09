import json
from datetime import datetime

from bson.objectid import ObjectId


def get_opinion_from_zcase(zcase):
    try:
        casebody = zcase.get('casebody')
        data = casebody.get('data')
        opinions = data.get('opinions')
        this_opinion = opinions[0]
        opinion_text = this_opinion.get('text')
        return opinion_text
    except Exception as e:
        return f"No opinion for ObjectId: {zcase.get('_id')}"


def convert_object_to_json(data):
    """
    Convert a potentially nested list (or any data structure) to a JSON string.
    Handles custom objects by attempting to convert them to dictionaries.
    """
    serializable_data = convert_object_to_serializable(data)
    this_json = json.dumps(serializable_data, indent=4)
    return json.loads(this_json)


def convert_object_to_serializable(obj):
    """
    Recursively convert objects in the data structure to JSON serializable types,
    handling MongoDB ObjectId, datetime objects, custom objects, tuples, and more.
    """
    if isinstance(obj, dict):
        return {key: convert_object_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_object_to_serializable(item) for item in obj]
    elif isinstance(obj, ObjectId):
        return str(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif hasattr(obj, "__dict__"):
        return {attr: convert_object_to_serializable(getattr(obj, attr)) for attr in dir(obj)
                if not attr.startswith(('__', '_')) and not callable(getattr(obj, attr))}
    else:
        # Fallback for other types that json.dumps cannot serialize directly
        try:
            # Attempt to use the default serializer; if this fails, convert to string
            return json.dumps(obj, default=str)
        except TypeError:
            return str(obj)