#
# robot.py
# Bart Trzynadlowski
#
# Robot controller. 
#

from enum import auto, Enum
from math import pi
from typing import List

from dynamixel_sdk import *
from ikpy.chain import Chain
import numpy as np

from .dynamixel import Dynamixel, ReadAttribute, OperatingMode


motor_ids = [ 1, 2, 3, 4, 5 ]
urdf_filepath = "robot/arm.urdf"

class Arm:
    class MotorControlType(Enum):
        PWM = auto()
        POSITION_CONTROL = auto()
        DISABLED = auto()
        UNKNOWN = auto()
        
    def __init__(self, port: str):
        self._chain = Chain.from_urdf_file(urdf_file=urdf_filepath, active_links_mask=[ False, False, True, True, False, True, True, False ])
    
        config = Dynamixel.Config(
            baudrate=57600,
            protocol_version=2,
            device_name=port
        )
        self._dynamixel = Dynamixel(config=config)

        self._position_reader = GroupSyncRead(
            port=self._dynamixel.portHandler,
            ph=self._dynamixel.packetHandler,
            start_address=ReadAttribute.POSITION.value,
            data_length=4
        )

        for id in motor_ids:
            self._position_reader.addParam(id)
        
        self._velocity_reader = GroupSyncRead(
            port=self._dynamixel.portHandler,
            ph=self._dynamixel.packetHandler,
            start_address=ReadAttribute.VELOCITY.value,
            data_length=4
        )

        for id in motor_ids:
            self._velocity_reader.addParam(id)

        self._position_writer = GroupSyncWrite(
            port=self._dynamixel.portHandler,
            ph=self._dynamixel.packetHandler,
            start_address=self._dynamixel.ADDR_GOAL_POSITION,
            data_length=4
        )

        for id in motor_ids:
            self._position_writer.addParam(id, [2048])

        self._pwm_writer = GroupSyncWrite(
            port=self._dynamixel.portHandler,
            ph=self._dynamixel.packetHandler,
            start_address=self._dynamixel.ADDR_GOAL_PWM,
            data_length=2
        )
        
        for id in motor_ids:
            self._pwm_writer.addParam(id, [2048])

        self._disable_torque()
        
    def __del__(self):
        self._disable_torque()
    
    def set_end_effector_target_position(self, position: List[float]):
        # Note: joints here are the URDF joints (as opposed to API's notion of joints, which are just
        # the motors)
        motor_id_by_joint_idx = { 2: 1, 3: 2, 5: 3, 6: 4 }
        joint_idx_by_motor_id = { motor_id: joint_idx for joint_idx, motor_id in motor_id_by_joint_idx.items() }

        # Target position from meters -> mm (units used in URDF)
        position_mm = [ position[0] * 1e3, position[1] * 1e3, position[2] * 1e3 ]

        # IK
        joint_radians = self._chain.inverse_kinematics(target_position=position_mm)
        motor_radians = []
        for motor_id in sorted(joint_idx_by_motor_id.keys()):
            joint_idx = joint_idx_by_motor_id[motor_id]
            radians = joint_radians[joint_idx]
            motor_radians.append(radians)
        motor_radians[1] = -motor_radians[1]

        # Temporary: gripper position to 0
        motor_radians.append(0)

        # Set position
        self.set_joint_goals(radians=motor_radians, wait=False)
    
    def read_joint_values(self, tries: int = 2) -> List[int]:
        """
        Reads the joint positions.
        
        Parameters
        ----------
        tries : int
            Maximum number of tries to read the position.
        
        Returns
        -------
        List[int]
            List of joint positions in range [0, 4096]. Center is 2048, 0 and 4096 are 180 degrees
            in each direction.
        """
        result = self.position_reader.txRxPacket()
        if result != 0:
            if tries > 0:
                return self._read_joint_positions(tries=tries - 1)
            else:
                print("Error: Failed to read joint positions")
        positions = []
        for id in self.servo_ids:
            position = self._position_reader.getData(id, ReadAttribute.POSITION.value, 4)
            if position > 2 ** 31:
                position -= 2 ** 32
            positions.append(position)
        return positions
    
    def set_joint_goals(self, wait: bool = False, **kwargs):
        if "radians" in kwargs and "degrees" in kwargs:
            raise TypeError("set_joint_goals(): Both 'radians' and 'degrees' cannot be specified simultaneously")
        positions = []
        if "radians" in kwargs:
            positions = [ self._radians_to_position(radians=radians) for radians in kwargs["radians"] ]
        elif "degrees" in kwargs:
            positions = [ self._degrees_to_position(degrees=degrees) for degrees in kwargs["degrees"] ]
        else:
            raise TypeError("set_joint_goals(): Specify either 'radians' or 'degrees' for each joint")
        self._set_joint_positions(positions=positions)
        if wait:
            self._wait_until_stopped()
        
    @staticmethod
    def _degrees_to_position(degrees: float) -> int:
        degrees = max(-180, min(180, degrees))
        abs_pos = round(abs(degrees) / 180 * 2048)
        return 2048 + (abs_pos if degrees >= 0 else -abs_pos)

    @staticmethod
    def _radians_to_position(radians: float) -> int:
        radians = max(-pi, min(pi, radians))
        abs_pos = round(abs(radians) / pi * 2048)
        return 2048 + (abs_pos if radians >= 0 else -abs_pos)

    def _set_joint_positions(self, positions: List[int]):
        """
        Sets joint positions (each [0,4096]).

        Parameters
        ----------
        positions : List[int]
            Position for each motor.
        """
        assert len(positions) == len(motor_ids)
        if not self._motor_control_state is Arm.MotorControlType.POSITION_CONTROL:
            self._set_position_control()
        for i, motor_id in enumerate(motor_ids):
            data_write = [
                DXL_LOBYTE(DXL_LOWORD(positions[i])),
                DXL_HIBYTE(DXL_LOWORD(positions[i])),
                DXL_LOBYTE(DXL_HIWORD(positions[i])),
                DXL_HIBYTE(DXL_HIWORD(positions[i]))
            ]
            self._position_writer.changeParam(motor_id, data_write)
        self._position_writer.txPacket()

    def _wait_until_stopped(self, timeout_seconds: float = 1):
        started_at = time.time()
        all_stopped = False
        while not all_stopped:
            if timeout_seconds > 0 and time.time() - started_at >= timeout_seconds:
                break
            all_stopped = True
            for i, motor_id in enumerate(motor_ids):
                all_stopped &= self._dynamixel.is_stopped(motor_id=motor_id)

    def _enable_torque(self):
        for motor_id in motor_ids:
            self._dynamixel._enable_torque(motor_id)
    
    def _disable_torque(self):
        self._motor_control_state = Arm.MotorControlType.DISABLED
        for motor_id in motor_ids:
            self._dynamixel._disable_torque(motor_id)
    
    def _set_pwm_control(self):
        self._disable_torque()
        for motor_id in motor_ids:
            self._dynamixel.set_operating_mode(motor_id, OperatingMode.PWM)
        self._enable_torque()
        self._motor_control_state = Arm.MotorControlType.PWM

    def _set_position_control(self):
        self._disable_torque()
        for motor_id in motor_ids:
            self._dynamixel.set_operating_mode(motor_id, OperatingMode.POSITION)
        self._enable_torque()
        self._motor_control_state = Arm.MotorControlType.POSITION_CONTROL
    

        



