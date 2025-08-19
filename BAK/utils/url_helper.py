from pydantic import AnyUrl, ValidationError
from typing import Any, Dict, Optional

class UrlHelper:
    """Centralized, robust URL utility for Pydantic, env, and Mongo pipelines."""

    @staticmethod
    def make_url(
        scheme: str = "http",
        host: str = "localhost",
        port: int = 80,
        path: str = "/"
    ) -> AnyUrl:
        """
        Build a Pydantic AnyUrl instance with the correct scheme.
        Ensures that schemes like 'ws' or 'wss' work properly.

        Example:
            UrlHelper.make_url("ws", "localhost", 8090, "/fleet/ws")
            → AnyUrl('ws://localhost:8090/fleet/ws')
        """
        return AnyUrl.build(scheme=scheme, host=host, port=port, path=path)

    @staticmethod
    def parse_url(val: Any) -> Optional[AnyUrl]:
        """Parse or validate a string as a Pydantic AnyUrl, return None if invalid."""
        if isinstance(val, AnyUrl):
            return val
        if isinstance(val, str):
            try:
                return AnyUrl(val)
            except (ValidationError, ValueError, TypeError):
                return None
        return None

    @classmethod
    def bulk_url_fields(
        cls,
        data: Dict[str, Any],
        url_fields: set = {"agent_url", "fleet_server_url"}
    ) -> Dict[str, Any]:
        """
        Returns a copy of data with all url_fields converted to AnyUrl if needed.
        Ignores fields not present, recurses into nested dicts/lists.
        """
        def convert(val):
            if isinstance(val, dict):
                return cls.bulk_url_fields(val, url_fields)
            elif isinstance(val, list):
                return [convert(i) for i in val]
            return val

        out = {}
        for k, v in data.items():
            if k in url_fields:
                parsed = cls.parse_url(v)
                out[k] = parsed if parsed is not None else v
            else:
                out[k] = convert(v)
        return out

# --- Example usage ---

# Create a URL from components
url = UrlHelper.make_url("ws", "localhost", 8090, "/fleet/ws")

# Parse a string or validate env var as AnyUrl
fleet_ws_url = UrlHelper.parse_url("ws://127.0.0.1:8090/fleet/ws")

# Convert dict fields to AnyUrl for SafeResult or Mongo output
agent_data = {
    "agent_url": "http://localhost:8000/",
    "fleet_server_url": "ws://localhost:8090/fleet/ws",
    "description": "Agent demo"
}
fixed = UrlHelper.bulk_url_fields(agent_data)
# Now fixed['agent_url'] and fixed['fleet_server_url'] are AnyUrl
