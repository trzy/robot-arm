#
# udp.py
# Bart Trzynadlowski
#
# Asynchronous UDP server. Designed to be run as an asynchronous task in an event loop.
#

import asyncio
import time
from typing import Any, Dict, Tuple

from pydantic import BaseModel

from .message_handling import MessageHandler
from .serialization import serialize
from .session import Session


def addr_to_endpoint(addr: Tuple[str | Any, int]) -> str:
    return f"{addr[0]}:{addr[1]}"

class UDPSession(Session):
    def __init__(self, transport: asyncio.DatagramTransport, address: Tuple[str | Any, int], message_handler: MessageHandler):
        super().__init__()
        self.remote_endpoint = f"udp://{addr_to_endpoint(addr=address)}"
        self._addr = address
        self._transport = transport
        self._message_handler = message_handler

    def __del__(self):
        print("UDPSession object destroyed")

    def __str__(self):
        return str(self.remote_endpoint)
    
    def is_reliable(self):
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
        try:
            json_string = serialize(message=message)
            json_bytes = json_string.encode("utf-8")
            total_size = 4 + len(json_bytes)  # size prefix + JSON payload
            message_bytes = (
                int(total_size).to_bytes(length=4, byteorder="little")
                + json_bytes
            )
            self._transport.sendto(data=message_bytes, addr=self._addr)
            #print("Sent %d bytes" % len(message_bytes))
        except Exception as e:
            # Connection error, most likely. Nothing we can do but swallow it and
            # hope it will be detected elsewhere. Makes for a more ergonomic API when
            # we don't have to worry about send() blowing up.
            print("Exception caught while trying to send: %s" % e)

class UDPReceiver(asyncio.DatagramProtocol):
    def __init__(self, message_handler: MessageHandler):
        super().__init__()
        self._message_handler = message_handler
        self._session_by_addr: Dict[Tuple[str | Any, int], UDPSession] = {}

    def connection_made(self, transport):
        self._transport = transport

    def datagram_received(self, data, addr):
        timestamp = time.time()

        # Map each endpoint to a session (currently these are never destroyed)
        session = self._session_by_addr.get(addr)
        if session is None:
            session = UDPSession(transport=self._transport, address=addr, message_handler=self._message_handler)
            self._session_by_addr[addr] = session
            asyncio.create_task(self._message_handler.on_connect(session=session))
        
        # Decode message and dispatch to message handler
        try:
            if len(data) < 4:
                raise ConnectionError(f"Received datagram with incomplete header ({len(data)} bytes but expected 4)")
            total_size = int.from_bytes(bytes=data[0:4], byteorder="little")
            payload_size = total_size - 4
            if payload_size > 0:
                payload = data[4:]
                if len(payload) != payload_size:
                    raise ConnectionError(f"Received message with incomplete body ({len(payload)} bytes but expected {payload_size})")
                json_string = payload.decode("utf-8")
                #print("Received %d bytes" % total_size)
                asyncio.create_task(self._message_handler.handle_message(session=session, json_string=json_string, timestamp=timestamp))
        except ConnectionError as e:
            print("Error from %s: %s" % (addr_to_endpoint(addr=addr), e))
        except Exception as e:
            print("Unexpected error in UDPReceiver: %s" % e)

class Server:
    def __init__(self, port: int, message_handler: MessageHandler):
        self._port = port
        self._message_handler = message_handler

    async def run(self):
        loop = asyncio.get_running_loop()
        try:
            transport, protocol = await loop.create_datagram_endpoint(
                protocol_factory=lambda: UDPReceiver(message_handler=self._message_handler),
                local_addr=("0.0.0.0", self._port) 
            )
            try:
                await asyncio.Future()
            finally:
                transport.close()
        except Exception as e:
            print("Error: %s" % e)
            raise
