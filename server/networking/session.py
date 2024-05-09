#
# session.py
# Bart Trzynadlowski
#
# Abstract connection session object.
#

from pydantic import BaseModel


class Session:
    def __str__(self):
        return "<Session>"
    
    def is_reliable(self) -> bool:
        return False

    async def send(self, message: BaseModel):
        """
        Sends an object as a JSON-encoded message using our custom framing protocol:

        Offset  Length  Description
        ------  ------  -----------
        0       4       Length of entire framed message, as a little-endian 32-
                        bit unsigned integer, including these 4 bytes: N.
        4       N - 4   Payload as encoded by LaserTagJSONEncoder. The length is
                        the length given at offset 0 less 4 bytes (i.e., the JSON
                        payload itself).

        Parameters
        ----------
        message : BaseModel
            A Pydantic BaseModel object that will automatically be serialized before transmission.
        """
        pass
