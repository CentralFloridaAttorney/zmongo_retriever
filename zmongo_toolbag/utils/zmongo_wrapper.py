from safe_result import SafeResult
import logging

log = logging.getLogger("mongo_utils")

def unwrap(result: SafeResult, *, quiet: bool = False):
    """
    Return result.original() on success or raise/return None on failure.
    This returns the original data with _id restored as ObjectId.
    """
    if result.success:
        return result.original()
    if quiet:
        log.error("Mongo error: %s", result.error)
        return None
    raise RuntimeError(result.error)
