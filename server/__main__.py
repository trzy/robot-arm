#
# __main__.py
# Bart Trzynadlowski
#
# Server main module. Listens for TCP connections from iPhone and controls the robot.
#

from annotated_types import Len
import argparse
import asyncio
import os
import platform
from typing import Annotated, Any, Dict, List, Tuple, Type

import cv2
import numpy as np
from pydantic import BaseModel

from .camera import CameraProcess, CameraFrameProvider
from .dataset import DatasetWriter
from .networking import Session, TCPServer, UDPServer, handler, MessageHandler
from .robot import serial_ports, find_serial_port, ArmProcess, ArmObservation
from .util import get_next_numbered_dirname


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
    def __init__(self, tcp_port: int, udp_port: int, arm_process: ArmProcess, camera_process: CameraProcess, recording_dir: str | None):
        super().__init__()
        self.sessions = set()
        self._tcp_server = TCPServer(port=tcp_port, message_handler=self)
        self._udp_server = UDPServer(port=udp_port, message_handler=self)
        self._arm_process = arm_process
        self._camera_process = camera_process
        self._recording_dir = recording_dir
        self._dataset_writer = self._new_dataset_writer()

        self._position = np.array([ 0, 5*2.54*1e-2, 9*2.54*1e-2 ])  # pretty close to 0 position
        arm_process.move_arm(
            position=self._position,
            gripper_open_amount=0,
            gripper_rotate_degrees=0
        )
        arm_process.set_camera_frame_provider(provider=CameraFrameProvider())

    def _new_dataset_writer(self):
        return DatasetWriter(recording_dir=self._recording_dir, dataset_prefix="example")
  
    def _record_observation(self, observation: ArmObservation):
        self._dataset_writer.record_observation(
            frame=observation.frame,
            observed_motor_radians=observation.observed_motor_radians,
            target_motor_radians=observation.target_motor_radians
        )

    def _finish_dataset(self):
        self._dataset_writer.finish()
        self._dataset_writer = self._new_dataset_writer()

    async def run(self):
        await asyncio.gather(self._tcp_server.run(), self._udp_server.run())
    
    async def on_connect(self, session: Session):
        print("Connection from: %s" % session.remote_endpoint)
        await session.send(HelloMessage(message = "Hello from Robophone Python server running on %s %s" % (platform.system(), platform.release())))
        self.sessions.add(session)
    
    async def on_disconnect(self, session: Session):
        print("Disconnected from: %s" % session.remote_endpoint)
        self.sessions.remove(session)
        self._finish_dataset()
    
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
            observation = self._arm_process.move_arm(
                position=position,
                gripper_open_amount=msg.gripperOpenAmount,
                gripper_rotate_degrees=msg.gripperRotateDegrees,
                wait_for_frame=True
            )
            self._record_observation(observation=observation)


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
    parser.add_argument("--list-ports", action="store_true", help="List available serial ports and exit")
    parser.add_argument("--port", metavar="name", action="store", type=str, help="Serial port to use")
    parser.add_argument("--camera", metavar="index", action="store", type=int, help="Camera to use")
    parser.add_argument("--record-to", metavar="directory", action="store", type=str, help="Save recorded data")
    options = parser.parse_args()

    port = get_serial_port()
    arm_process = ArmProcess(serial_port=port)
    camera_process = CameraProcess(camera_idx=options.camera)

    tasks = []
    server = RobotArmServer(tcp_port=8000, udp_port=8001, arm_process=arm_process, camera_process=camera_process, recording_dir=options.record_to)
    loop = asyncio.new_event_loop()
    tasks.append(loop.create_task(server.run()))
    try:
        loop.run_until_complete(asyncio.gather(*tasks))
    except asyncio.exceptions.CancelledError:
        print("\nExited normally")
    except:
        print("\nExited due to uncaught exception")
