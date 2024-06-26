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
from typing import Callable, Dict, List, Type

import numpy as np

from .arm import Arm
from ..camera import CameraFrameProvider


####################################################################################################
# Inter-Process Communication
#
# Internal objects passed back and forth between the server process and robot sub-process. Input and
# output queues are used for communication.
####################################################################################################

@dataclass
class FrameProviderCommand:
    provider: CameraFrameProvider

@dataclass
class ResetPoseCommand:
    wait_for_completion: bool = False

@dataclass
class MoveArmCommand:
    position: np.ndarray
    gripper_open_amount: float
    gripper_rotate_degrees: float

@dataclass
class SetMotorsCommand:
    target_motor_radians: List[float]

@dataclass
class GetObservationCommand:
    pass

@dataclass
class TerminateProcessCommand:
    pass


####################################################################################################
# Sub-Process Public API
####################################################################################################

_frame_provider: CameraFrameProvider | None = None

@dataclass
class CommandFinishedResponse:
    error: str | None = None
    frame: np.ndarray | None = None # shape (N,480,640,3)
    observed_motor_radians: List[float] | None = None
    target_motor_radians: List[float] | None = None

    def succeeded(self):
        return self.error is None

@dataclass
class ArmObservation:
    frame: np.ndarray
    observed_motor_radians: List[float]
    target_motor_radians: List[float]

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
            pass

        # Still busy?
        return self._num_commands_in_progress > 0

    def wait_until_not_busy(self):
        while self.is_busy():
            pass

    def reset_pose(self, wait_for_completion: bool = False):
        self._command_queue.put(ResetPoseCommand(wait_for_completion=wait_for_completion))
        self._num_commands_in_progress += 1

    def set_camera_frame_provider(self, provider: CameraFrameProvider, wait_for_completion: bool = True):
        self._command_queue.put(FrameProviderCommand(provider=provider))
        self._num_commands_in_progress += 1
        if wait_for_completion:
            self.wait_until_not_busy()

    def move_arm(self, position: np.ndarray, gripper_open_amount: float, gripper_rotate_degrees: float, wait_for_frame: bool = False) -> ArmObservation | None:
        self._command_queue.put(MoveArmCommand(position=position, gripper_open_amount=gripper_open_amount, gripper_rotate_degrees=gripper_rotate_degrees))
        self._num_commands_in_progress += 1
        if wait_for_frame:
            while True:
                response = self._try_get_response()
                if response is not None:
                    if response.frame is not None and response.observed_motor_radians is not None and response.target_motor_radians is not None:
                        return ArmObservation(frame=response.frame, observed_motor_radians=response.observed_motor_radians, target_motor_radians=response.target_motor_radians)
                    else:
                        break
        return None

    def set_motor_radians(self, target_motor_radians: List[float], wait_for_completion: bool = False) -> ArmObservation | None:
        self._command_queue.put(SetMotorsCommand(target_motor_radians=target_motor_radians))
        self._num_commands_in_progress += 1
        if wait_for_completion:
            while True:
                response = self._try_get_response()
                if response is not None:
                    if response.frame is not None and response.observed_motor_radians is not None and response.target_motor_radians is not None:
                        return ArmObservation(frame=response.frame, observed_motor_radians=response.observed_motor_radians, target_motor_radians=response.target_motor_radians)
                    else:
                        break
        return None

    def get_observation(self) -> ArmObservation | None:
        self._command_queue.put(GetObservationCommand())
        self._num_commands_in_progress += 1
        while True:
            response = self._try_get_response()
            if response is not None:
                assert response.frame is not None
                assert response.observed_motor_radians is not None
                return ArmObservation(frame=response.frame, observed_motor_radians=response.observed_motor_radians, target_motor_radians=[])

    def _try_get_response(self) -> CommandFinishedResponse | None:
        try:
            response = self._response_queue.get_nowait()
            self._num_commands_in_progress -= 1
            return response
        except Empty:
            return None

    def _run(command_queue: Queue, response_queue: Queue, serial_port: str):
        arm = Arm(port=serial_port)
        handler_by_command: Dict[Type, Callable] = {
            FrameProviderCommand: ArmProcess._handle_frame_provider,
            ResetPoseCommand: ArmProcess._handle_reset_position,
            MoveArmCommand: ArmProcess._handle_move_arm,
            SetMotorsCommand: ArmProcess._handle_set_motors,
            GetObservationCommand: ArmProcess._handle_get_observation
        }
        while True:
            command = command_queue.get()
            if isinstance(command, TerminateProcessCommand):
                break
            else:
                handler = handler_by_command.get(type(command))
                if handler:
                    handler(arm, command, response_queue)

    def _handle_frame_provider(arm: Arm, command: FrameProviderCommand, response_queue: Queue):
        global _frame_provider
        _frame_provider = command.provider
        response_queue.put(CommandFinishedResponse())

    def _handle_reset_position(arm: Arm, command: ResetPoseCommand, response_queue: Queue):
        arm.set_motor_goals(degrees=[0,0,0,0,0], wait=command.wait_for_completion)
        response_queue.put(CommandFinishedResponse())

    def _handle_move_arm(arm: Arm, command: MoveArmCommand, response_queue: Queue):
        response = CommandFinishedResponse()

        # Clamp vertical position so we don't hit table.
        # Note: Kinematic chain seems to be set up wrong because 0.0409m is the height of motor 1
        # axis above origin point. If chain were set up correctly, 0 would be the table surface
        # position!
        position = command.position
        position[2] = max(0.0409, position[2])

        # Get current joint angles
        current_motor_radians = arm.read_motor_radians()

        # Get current frame
        global _frame_provider
        if _frame_provider:
            response.frame = _frame_provider.get_frame_buffer()

        # Gripper open amount
        closed_degrees = -5.0
        open_degrees = 90.0
        gripper_open_degrees = min(1, max(0, command.gripper_open_amount)) * (open_degrees - closed_degrees) + closed_degrees

        # Gripper rotation
        gripper_rotate_degrees = min(90.0, max(-90.0, command.gripper_rotate_degrees))

        # Apply to arm
        target_motor_radians = arm.set_end_effector_target_position(target_position=position, initial_motor_radians=current_motor_radians, gripper_open_degrees=gripper_open_degrees, gripper_rotate_degrees=gripper_rotate_degrees)

        # Write current motor angles (i.e., the follower observation) and the target motor angles
        # (i.e., the leader actions that should be predicted) to response
        response.observed_motor_radians = current_motor_radians
        response.target_motor_radians = target_motor_radians
        response_queue.put(response)

    def _handle_set_motors(arm: Arm, command: SetMotorsCommand, response_queue: Queue):
        global _frame_provider
        response = CommandFinishedResponse()
        if _frame_provider:
            response.frame = _frame_provider.get_frame_buffer()
        response.target_motor_radians = command.target_motor_radians
        arm.set_motor_goals(radians=command.target_motor_radians)
        #TODO: how long to wait?
        response.observed_motor_radians = arm.read_motor_radians()
        response_queue.put(response)

    def _handle_get_observation(arm: Arm, command: GetObservationCommand, response_queue: Queue):
        global _frame_provider
        response = CommandFinishedResponse()
        if _frame_provider:
            response.frame = _frame_provider.get_frame_buffer()
        response.observed_motor_radians = arm.read_motor_radians()
        response_queue.put(response)