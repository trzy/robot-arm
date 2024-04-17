#
# serialization.py
# Bart Trzynadlowski
#
# Serializes Pydantic BaseModel objects into messages (JSON with an "__id" field added set to the 
# class name).
#
# For example:
#
#   HelloMessage(BaseModel):
#       message: str
#
# Serializes to:
#
#   { "__id": "HelloMessage", "message": "Message here." }
#

import json
from typing import Any, Dict, Type

from pydantic import BaseModel


def serialize(message: BaseModel) -> str:
    dictionary = message.model_dump()
    dictionary["__id"] = message.__class__.__name__
    return json.dumps(dictionary)

def deserialize(message_type: Type[BaseModel], dictionary: Dict[str, Any]) -> Any:
    return message_type.parse_obj(dictionary)