#
# process.py
# Bart Trzynadlowski
#
# Robot arm process. Exists as separate process to allow concurrent processing (e.g., network
# message reception) to continue.
#

from dataclasses import dataclass
from multiprocessing import Queue, Process
from queue import Empty
import time
from typing import Callable, Dict, Type

import cv2
import numpy as np

from .arm import Arm
from ..util import FrameRateCalculator


####################################################################################################
# Inter-Process Communication
#
# Internal objects passed back and forth between the server process and robot sub-process. Input and
# output queues are used for communication.
####################################################################################################

@dataclass
class ResetPoseCommand:
    wait_for_completion: bool = False

@dataclass
class MoveArmCommand:
    position: np.ndarray
    gripper_open_amount: float
    gripper_rotate_degrees: float

@dataclass
class TerminateProcessCommand:
    pass


####################################################################################################
# Sub-Process Public API
####################################################################################################

from ..camera import CameraProcess
_camera_process = None

@dataclass
class CommandFinishedResponse:
    error: str | None = None

    def succeeded(self):
        return self.error is None
    
class ArmProcess:
    def __init__(self, serial_port: str):
        self._command_queue = Queue()
        self._response_queue = Queue()
        
        print("Starting robot arm process...")
        process_args = (
            self._command_queue,
            self._response_queue,
            serial_port
        )
        self._process = Process(target=ArmProcess._run, args=process_args)
        self._process.start()
        self._num_commands_in_progress = 0
    
    def __del__(self):
        self._command_queue.put(TerminateProcessCommand())
        self._process.join()
        print("Terminated robot arm process")

    def is_busy(self):
        # Purge response queue of any completed responses
        while self._try_get_response() is not None:
            self._num_commands_in_progress -= 1

        # Still busy?
        return self._num_commands_in_progress > 0
    
    def reset_pose(self, wait_for_completion: bool = False):
        self._command_queue.put(ResetPoseCommand(wait_for_completion=wait_for_completion))
        self._num_commands_in_progress += 1
    
    def move_arm(self, position: np.ndarray, gripper_open_amount: float, gripper_rotate_degrees: float):
        self._command_queue.put(MoveArmCommand(position=position, gripper_open_amount=gripper_open_amount, gripper_rotate_degrees=gripper_rotate_degrees))

    def _try_get_response(self) -> CommandFinishedResponse | None:
        try:
            return self._response_queue.get_nowait()
        except Empty:
            return None

    def _run(command_queue: Queue, response_queue: Queue, serial_port: str):
        global _camera_process
        _camera_process = CameraProcess(camera_idx=0)
        arm = Arm(port=serial_port)
        handler_by_command: Dict[Type, Callable] = {
            ResetPoseCommand: ArmProcess._handle_reset_position,
            MoveArmCommand: ArmProcess._handle_move_arm
        }
        while True:
            command = command_queue.get()
            if isinstance(command, TerminateProcessCommand):
                break
            else:
                handler = handler_by_command.get(type(command))
                if handler:
                    handler(arm, command, response_queue)

    def _handle_reset_position(arm: Arm, command: ResetPoseCommand, response_queue: Queue):
        arm.set_motor_goals(degrees=[0,0,0,0,0], wait=command.wait_for_completion)
        response_queue.put(CommandFinishedResponse())
    
    def _handle_move_arm(arm: Arm, command: MoveArmCommand, response_queue: Queue):
        # Clamp vertical position so we don't hit table. 
        # Note: Kinematic chain seems to be set up wrong because 0.0409m is the height of motor 1
        # axis above origin point. If chain were set up correctly, 0 would be the table surface
        # position!
        position = command.position
        position[2] = max(0.0409, position[2])

        # Get current joint angles
        motor_radians = arm.read_motor_radians()

        # Gripper open amount
        closed_degrees = -5.0
        open_degrees = 90.0
        gripper_open_degrees = min(1, max(0, command.gripper_open_amount)) * (open_degrees - closed_degrees) + closed_degrees

        # Gripper rotation
        gripper_rotate_degrees = min(90.0, max(-90.0, command.gripper_rotate_degrees))

        # Get camera frame
        # t0 = time.perf_counter()
        # success, frame = _camera.read()
        # t1 = time.perf_counter()
        # _fps.record_frame()
        # if not success:
        #     print("Camera error")
        # t0 = time.perf_counter()
        # cv2.putText(frame, "%1.1f" % _fps.fps, org = (50, 50), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, color = (0, 255, 255), thickness = 2, lineType = cv2.LINE_AA)
        # cv2.imshow("Camera", frame)
        # cv2.waitKey(1)
        # t1 = time.perf_counter()
        # print(f"{(t1-t0)/1e-3} ms")
        global _camera_process
        cv2.imshow("Camera", _camera_process.get_frame_buffer())
        cv2.waitKey(1)
        

        # Apply to arm
        arm.set_end_effector_target_position(target_position=position, initial_motor_radians=motor_radians, gripper_open_degrees=gripper_open_degrees, gripper_rotate_degrees=gripper_rotate_degrees)
        response_queue.put(CommandFinishedResponse())