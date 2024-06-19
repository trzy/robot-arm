#
# __main__.py
# Bart Trzynadlowski
#
# Server main module. Listens for TCP and UDP connections from iPhone and controls the robot,
# records example episodes, and performs inference if requested.
#
# Usage:
#
#   - To discover serial ports on which robot may be connected:
#
#       python -m server --list-serial-ports
#
#   - Serial ports can be specified with wildcards. To e.g. connect to the first connected USB port
#     on macOS:
#
#       python -m server --serial-port=*usb*
#
#   - Recording episodes to the "cube" subdirectory and using a different camera:
#
#       python -m server --serial-port=COM4 --camera=1 --record-to=cube
#
#   - Replay captured actions from a particular episode to the robot:
#
#       python -m server --serial-port=COM4 --replay-from=cube/example-0/data.hdf5
#
#   - Perform inference:
#
#       python -m server --serial-port=COM4 --infer
#

from annotated_types import Len
import argparse
import asyncio
import base64
import platform
from typing import Annotated, List, Optional

import cv2
import numpy as np
from pydantic import BaseModel

from .camera import CameraProcess
from .dataset import DatasetWriter, read_dataset
from .networking import Session, TCPClient, TCPServer, UDPServer, handler, MessageHandler
from .robot import serial_ports, find_serial_port, ArmProcess, ArmObservation


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

class BeginEpisodeMessage(BaseModel):
    unused: Optional[int | None] = None # dummy field because Pydantic doesn't support empty models

class EndEpisodeMessage(BaseModel):
    unused: Optional[int | None] = None

class InferenceRequestMessage(BaseModel):
    motor_radians: Annotated[List[float], Len(min_length=5, max_length=5)]
    frame: str

class InferenceResponseMessage(BaseModel):
    target_motor_radians: Annotated[List[float], Len(min_length=5, max_length=5)]


####################################################################################################
# Inference Client
#
# Communicates with inference server.
####################################################################################################

class InferenceClient(MessageHandler):
    def __init__(self, queue: asyncio.Queue, endpoint: str):
        super().__init__()
        self._client: TCPClient | None
        self._endpoint = endpoint
        self._session: Session | None = None    # when connected, session to send on
        self._queue = queue

    def is_connected(self):
        return self._session is not None

    async def run(self):
        while True:
            try:
                self._client = TCPClient(connect_to=self._endpoint, message_handler=self)
                await self._client.run()
                await asyncio.sleep(5)
                print("Reconnecting to inference server...")
            except Exception as e:
                print(f"Error in InferenceClient: {e}")

    async def send_observation(self, observation: ArmObservation):
        if self._session is not None:
            success, jpeg = cv2.imencode('.jpg', observation.frame)
            jpeg_bytes = jpeg.tobytes() if success else bytes()
            frame_base64 = base64.b64encode(jpeg_bytes)
            msg = InferenceRequestMessage(motor_radians=observation.observed_motor_radians, frame=frame_base64)
            await self._session.send(message=msg)
            print("Sent inference request")

    async def on_connect(self, session: Session):
        print("Connected to inference server: %s" % session.remote_endpoint)
        await session.send(HelloMessage(message = "Hello from Robophone Python server running on %s %s" % (platform.system(), platform.release())))
        self._session = session

    async def on_disconnect(self, session: Session):
        print("Disconnected from inference server: %s" % session.remote_endpoint)
        self._session = None
        await self._queue.put(None) # send a None to indicate disconnected

    @handler(HelloMessage)
    async def handle_HelloMessage(self, session: Session, msg: HelloMessage, timestamp: float):
        print("Hello received: %s" % msg.message)

    @handler(InferenceResponseMessage)
    async def handle_InferenceResponseMessage(self, session: Session, msg: InferenceResponseMessage, timestamp: float):
        await self._queue.put(msg)


####################################################################################################
# Server
#
# Listens for connections and responds to messages from the iPhone app.
####################################################################################################

class RobotArmServer(MessageHandler):
    def __init__(
        self,
        tcp_port: int,
        udp_port: int,
        arm_process: ArmProcess,
        camera_process: CameraProcess,
        infer: bool,
        inference_endpoint: str,
        recording_dir: str | None,
        replay_filepath: str | None,
        replay_hz: float,
        replay_observed: bool,
        infer_on_replay: bool
    ):
        super().__init__()
        self.sessions = set()
        self._inference_queue = asyncio.Queue()
        self._inference_client = InferenceClient(queue=self._inference_queue, endpoint=inference_endpoint) if infer else None
        self._tcp_server = TCPServer(port=tcp_port, message_handler=self)
        self._udp_server = UDPServer(port=udp_port, message_handler=self)
        self._arm_process = arm_process
        self._camera_process = camera_process
        self._recording_dir = recording_dir
        self._dataset_writer = None
        self._replay_filepath = replay_filepath
        self._replay_hz = replay_hz
        self._replay_observed = replay_observed
        self._infer_on_replay = infer_on_replay

        self._position = np.array([ 0, 5*2.54*1e-2, 9*2.54*1e-2 ])  # pretty close to 0 position
        arm_process.move_arm(
            position=self._position,
            gripper_open_amount=0,
            gripper_rotate_degrees=0,
            wait_for_frame=True # wait until completion
        )
        arm_process.set_camera_frame_provider(provider=camera_process.create_frame_provider())

    def _new_dataset_writer(self):
        return DatasetWriter(recording_dir=self._recording_dir, dataset_prefix="example", num_cameras=self._camera_process.num_cameras)

    def _record_observation(self, observation: ArmObservation):
        if self._dataset_writer is not None:
            self._dataset_writer.record_observation(
                frame=observation.frame,
                observed_motor_radians=observation.observed_motor_radians,
                target_motor_radians=observation.target_motor_radians
            )

    def _finish_dataset(self):
        self._dataset_writer.finish()
        self._dataset_writer = None

    async def run(self):
        tasks = [ self._tcp_server.run(), self._udp_server.run(), self._run_replay_and_inference() ]
        if self._inference_client is not None:
            tasks.append(self._inference_client.run())
        await asyncio.gather(*tasks)

    async def _run_replay_and_inference(self):
        try:
            # First, handle replay to robot
            if self._replay_filepath and self._replay_hz > 0 and not self._infer_on_replay:
                await self._run_replay()

            # Next, perform inference
            if self._inference_client is None:
                return
            await asyncio.sleep(3)
            if self._infer_on_replay:
                # Perform inference on replay data
                if not (self._replay_filepath and self._replay_hz > 0):
                    return
                dataset = read_dataset(filepath=self._replay_filepath)
                print(f"Loaded replay data from {self._replay_filepath}")
                print(f"Replaying and inferring...")
                num_samples = len(dataset.frames)
                for i in range(num_samples):
                    if self._inference_client.is_connected():
                        observation = ArmObservation(
                            frame=dataset.frames[i],
                            observed_motor_radians=dataset.observed_motor_radians[i],
                            target_motor_radians=dataset.target_motor_radians[i]
                        )
                        await self._inference_client.send_observation(observation=observation)
                        msg: InferenceResponseMessage | None = await self._inference_queue.get()
                        if msg is not None:
                            self._arm_process.set_motor_radians(target_motor_radians=msg.target_motor_radians)
                    await asyncio.sleep(0.1)
                print("Finished replaying")
            else:
                # Perform inference on live input
                while True:
                    if self._inference_client.is_connected():
                        observation = self._arm_process.get_observation()
                        await self._inference_client.send_observation(observation=observation)
                        msg: InferenceResponseMessage | None = await self._inference_queue.get()
                        if msg is not None:
                            self._arm_process.set_motor_radians(target_motor_radians=msg.target_motor_radians)
                    await asyncio.sleep(0.1 / 2)
        except Exception as e:
            print(f"Error in replay/inference loop: {e}")

    async def _run_replay(self):
        dataset = read_dataset(filepath=self._replay_filepath)
        print(f"Loaded replay data from {self._replay_filepath}")
        print(f"Replaying at {self._replay_hz} Hz...")
        num_samples = len(dataset.frames)
        for i in range(num_samples):
            motor_radians = dataset.observed_motor_radians[i] if self._replay_observed else dataset.target_motor_radians[i]
            self._arm_process.set_motor_radians(target_motor_radians=motor_radians)
            await asyncio.sleep(1.0 / self._replay_hz)
        print("Finished replay")

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

    @handler(BeginEpisodeMessage)
    async def handle_BeginEpisodeMessage(self, session: Session, msg: BeginEpisodeMessage, timestamp: float):
        if self._recording_dir is not None:
            if self._dataset_writer is None:
                self._dataset_writer = self._new_dataset_writer()
                print(f"Begin episode: {self._dataset_writer.directory}")

    @handler(EndEpisodeMessage)
    async def handle_EndEpisodeMessage(self, session: Session, msg: EndEpisodeMessage, timestamp: float):
        if self._dataset_writer is not None:
            self._dataset_writer.finish()
            self._dataset_writer = None
            print("End episode")

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
    if options.list_serial_ports:
        print("\n".join(ports))
        exit()
    port = ports[0] if options.serial_port is None else find_serial_port(port_pattern=options.serial_port)
    if port is None:
        exit()
    print(f"Serial port: {port}")
    return port

if __name__ == "__main__":
    parser = argparse.ArgumentParser("robotest")
    parser.add_argument("--list-serial-ports", action="store_true", help="List available serial ports and exit")
    parser.add_argument("--serial-port", metavar="name", action="store", type=str, help="Serial port to use")
    parser.add_argument("--camera", metavar="indices", action="store", type=str, default="0", help="Camera(s) to use")
    parser.add_argument("--server-port", metavar="number", action="store", type=int, default=8000, help="Server port (for both TCP and UDP)")
    parser.add_argument("--record-to", metavar="directory", action="store", type=str, help="Save recorded data")
    parser.add_argument("--infer", action="store_true", help="Run inference")
    parser.add_argument("--inference-endpoint", action="store", type=str, default="localhost:8001", help="Endpoint (host:port) of inference server")
    parser.add_argument("--replay-from", metavar="file", type=str, help="Replay an episode captured in an hdf5 file")
    parser.add_argument("--replay-rate", metavar="hz", type=float, default=20, help="Rate (Hz) to replay episode at")
    parser.add_argument("--replay-observed", action="store_true", help="Replay observed angles instead of commanded angles")
    parser.add_argument("--infer-on-replay", action="store_true", help="Replay to inference server")
    options = parser.parse_args()

    if options.infer_on_replay:
        options.infer = True

    camera_idxs = [ int(camera_idx) for camera_idx in options.camera.split(",") ]

    serial_port = get_serial_port()
    arm_process = ArmProcess(serial_port=serial_port)
    camera_process = CameraProcess(camera_idxs=camera_idxs)

    tasks = []
    server = RobotArmServer(
        tcp_port=options.server_port,
        udp_port=options.server_port,
        arm_process=arm_process,
        camera_process=camera_process,
        infer=options.infer,
        inference_endpoint=options.inference_endpoint,
        recording_dir=options.record_to,
        replay_filepath=options.replay_from,
        replay_hz=options.replay_rate,
        replay_observed=options.replay_observed,
        infer_on_replay=options.infer_on_replay == True
    )
    loop = asyncio.new_event_loop()
    tasks.append(loop.create_task(server.run()))
    try:
        loop.run_until_complete(asyncio.gather(*tasks))
    except asyncio.exceptions.CancelledError:
        print("\nExited normally")
    except:
        print("\nExited due to uncaught exception")
