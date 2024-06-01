import constants
from model.data import ModelCriteria
import base64
import hashlib


def get_model_criteria(block: int) -> ModelCriteria:
    """Returns the model criteria at block."""
    criteria = None
    for b, crit in constants.MODEL_CRITERIA_BY_BLOCK:
        if block >= b:
            criteria = crit
    assert criteria is not None, f"No model criteria found for block {block}"
    return criteria


def get_hash_of_two_strings(string1: str, string2: str) -> str:
    """Hashes two strings together and returns the result."""

    string_hash = hashlib.sha256((string1 + string2).encode())

    return base64.b64encode(string_hash.digest()).decode("utf-8")
