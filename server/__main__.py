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

from .camera import CameraProcess
from .networking import Server, Session, handler, MessageHandler
from .robot import serial_ports, find_serial_port, ArmProcess


####################################################################################################
# Messages
#
# Keep these in sync with the corresponding messages in the iOS app.
####################################################################################################

class HelloMessage(BaseModel):
    message: str

class PoseStateMessage(BaseModel):
    gripperDeltaPosition: Annotated[List[float], Len(min_length=3, max_length=3)]
    gripperOpenAmount: float
    gripperRotateDegrees: float


####################################################################################################
# Server
#
# Listens for connections and responds to messages from the iPhone app.
####################################################################################################

class RobotArmServer(MessageHandler):
    def __init__(self, port: int, arm_process: ArmProcess, camera_process: CameraProcess):
        super().__init__()
        self.sessions = set()
        self._server = Server(port=port, message_handler=self)
        self._arm_process = arm_process
        self._camera_process = camera_process
        self._position = np.array([ 0, 5*2.54*1e-2, 9*2.54*1e-2 ])  # pretty close to 0 position
        arm_process.move_arm(
            position=self._position,
            gripper_open_amount=0,
            gripper_rotate_degrees=0,
            frame_grabber=None
        )
    
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
    
    @handler(PoseStateMessage)
    async def handle_PoseStateMessage(self, session: Session, msg: PoseStateMessage, timestamp: float):
        # Convert (x,y,z) from ARKit -> (x,z,y) in robot frame
        x = msg.gripperDeltaPosition[0]    # robot X axis is to the right
        y = -msg.gripperDeltaPosition[2]   # robot Y axis is in front
        z = msg.gripperDeltaPosition[1]    # robot Z axis is up
        delta_position = np.array([ x, y, z ])

        # Move arm if it is not busy
        if not self._arm_process.is_busy():
            position = self._position + delta_position
            self._arm_process.move_arm(
                position=position,
                gripper_open_amount=msg.gripperOpenAmount,
                gripper_rotate_degrees=msg.gripperRotateDegrees,
                frame_grabber=self._camera_process.frame_grabber
            )


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
    parser.add_argument("--port", metavar="name", action="store", type=str, help="Serial port to use")
    parser.add_argument("--camera", metavar="index", action="store", type=int, help="Camera to use")
    options = parser.parse_args()

    port = get_serial_port()
    arm_process = ArmProcess(serial_port=port)
    camera_process = CameraProcess(camera_idx=options.camera)

    tasks = []
    server = RobotArmServer(port=8000, arm_process=arm_process, camera_process=camera_process)
    loop = asyncio.new_event_loop()
    tasks.append(loop.create_task(server.run()))
    try:
        loop.run_until_complete(asyncio.gather(*tasks))
    except asyncio.exceptions.CancelledError:
        print("\nExited normally")
    except:
        print("\nExited due to uncaught exception")
