#
# __main__.py
# Bart Trzynadlowski
#
# Server main module. Listens for TCP connections from iPhone and controls the robot.
#

import asyncio
from dataclasses import dataclass
import platform
import sys
from typing import Any, Dict, List, Tuple, Type

from pydantic import BaseModel

from .networking import Server, Session, handler, MessageHandler


####################################################################################################
# Messages
#
# Keep these in sync with the corresponding messages in the iOS app.
####################################################################################################

class HelloMessage(BaseModel):
    message: str

class TransformMessage(BaseModel):
    matrix: List[float]


####################################################################################################
# Server
#
# Listens for connections and responds to messages from the iPhone app.
####################################################################################################

class RobotArmServer(MessageHandler):
    def __init__(self, port: int):
        super().__init__()
        self.sessions = set()
        self._server = Server(port=port, message_handler=self)
    
    async def run(self):
        await self._server.run()
    
    async def on_connect(self, session: Session):
        print("Connection from: %s" % session.remote_endpoint)
        await session.send(HelloMessage(message = "Hello from Robophone Python server running on %s %s" % (platform.system(), platform.release())))
        self.sessions.add(session)
    
    async def on_disconnect(self, session: Session):
        print("Disconnected from: %s" % session.remote_endpoint)
        self.sessions.remove(session)
    
    @handler(HelloMessage)
    async def handle_HelloMessage(self, session: Session, msg: HelloMessage, timestamp: float):
        print("Hello received: %s" % msg.message)
    
    @handler(TransformMessage)
    async def handle_TransformMessage(self, session: Session, msg: TransformMessage, timestamp: float):
        print(f"Transform received: {msg.matrix}")


####################################################################################################
# Program Entry Point
####################################################################################################

if __name__ == "__main__":
    tasks = []
    server = RobotArmServer(port=8000)
    loop = asyncio.new_event_loop()
    tasks.append(loop.create_task(server.run()))
    try:
        loop.run_until_complete(asyncio.gather(*tasks))
    except asyncio.exceptions.CancelledError:
        print("\nExited normally")
    except:
        print("\nExited due to uncaught exception")
