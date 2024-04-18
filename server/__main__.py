#
# __main__.py
# Bart Trzynadlowski
#
# Server main module. Listens for TCP connections from iPhone and controls the robot.
#

from annotated_types import Len
import argparse
import asyncio
import platform
from typing import Annotated, Any, Dict, List, Tuple, Type

import numpy as np
from pydantic import BaseModel

from .networking import Server, Session, handler, MessageHandler
from .robot import serial_ports, find_serial_port, Arm


####################################################################################################
# Messages
#
# Keep these in sync with the corresponding messages in the iOS app.
####################################################################################################

class HelloMessage(BaseModel):
    message: str

class PoseUpdateMessage(BaseModel):
    initialPose: Annotated[List[float], Len(min_length=16, max_length=16)]
    pose: Annotated[List[float], Len(min_length=16, max_length=16)]
    deltaPosition: Annotated[List[float], Len(min_length=3, max_length=3)]


####################################################################################################
# Server
#
# Listens for connections and responds to messages from the iPhone app.
####################################################################################################

class RobotArmServer(MessageHandler):
    def __init__(self, port: int, arm: Arm):
        super().__init__()
        self.sessions = set()
        self._server = Server(port=port, message_handler=self)
        self._arm = arm
    
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
    
    @handler(PoseUpdateMessage)
    async def handle_PoseUpdateMessage(self, session: Session, msg: PoseUpdateMessage, timestamp: float):
        #print(f"Pose received: {msg}")
        origin_position = np.array([ 0, 5*2.54*1e-2, 5*2.54*1e-2 ])
        position = origin_position + np.array(msg.deltaPosition)
        self._arm.set_end_effector_target_position(position=position)



####################################################################################################
# Program Entry Point
####################################################################################################

def get_serial_port() -> str:
    ports = serial_ports()
    if len(ports) == 0:
        print("No serial ports")
        exit()
    if options.list_ports:
        print("\n".join(ports))
        exit()
    port = ports[0] if options.port is None else find_serial_port(port_pattern=options.port)
    if port is None:
        exit()
    print(f"Serial port: {port}")
    return port

if __name__ == "__main__":
    parser = argparse.ArgumentParser("robotest")
    parser.add_argument("--list-ports", action="store_true", help="List available serial ports")
    parser.add_argument("--port", action="store", type=str, help="Serial port to use")
    options = parser.parse_args()

    port = get_serial_port()
    arm = Arm(port=port)
#    arm.set_joint_goals(degrees=[0,0,0,0,0])
    arm.set_end_effector_target_position(position=[0, 5*2.54*1e-2, 5*2.54*1e-2 ])

    tasks = []
    server = RobotArmServer(port=8000, arm=arm)
    loop = asyncio.new_event_loop()
    tasks.append(loop.create_task(server.run()))
    try:
        loop.run_until_complete(asyncio.gather(*tasks))
    except asyncio.exceptions.CancelledError:
        print("\nExited normally")
    except:
        print("\nExited due to uncaught exception")
