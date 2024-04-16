from __future__ import annotations
import argparse
from dataclasses import dataclass
import enum
from enum import Enum, auto
import glob
import math
import os
import sys
from typing import List, Union

from dynamixel_sdk import *  # Uses Dynamixel SDK library
from ikpy.chain import Chain
import numpy as np
import serial


class ReadAttribute(enum.Enum):
    TEMPERATURE = 146
    VOLTAGE = 145
    VELOCITY = 128
    POSITION = 132
    CURRENT = 126
    PWM = 124
    HARDWARE_ERROR_STATUS = 70
    HOMING_OFFSET = 20
    BAUDRATE = 8


class OperatingMode(enum.Enum):
    VELOCITY = 1
    POSITION = 3
    CURRENT_CONTROLLED_POSITION = 5
    PWM = 16
    UNKNOWN = -1


class Dynamixel:
    ADDR_TORQUE_ENABLE = 64
    ADDR_GOAL_POSITION = 116
    ADDR_VELOCITY_LIMIT = 44
    ADDR_GOAL_PWM = 100
    OPERATING_MODE_ADDR = 11
    POSITION_I = 82
    POSITION_P = 84
    ADDR_ID = 7

    @dataclass
    class Config:
        def instantiate(self): return Dynamixel(self)

        baudrate: int = 57600
        protocol_version: float = 2.0
        device_name: str = ''  # /dev/tty.usbserial-1120'
        dynamixel_id: int = 1

    def __init__(self, config: Config):
        self.config = config
        self.connect()

    def connect(self):
        if self.config.device_name == '':
            for port_name in os.listdir('/dev'):
                if 'ttyUSB' in port_name or 'ttyACM' in port_name:
                    self.config.device_name = '/dev/' + port_name
                    print(f'using device {self.config.device_name}')
        self.portHandler = PortHandler(self.config.device_name)
        # self.portHandler.LA
        self.packetHandler = PacketHandler(self.config.protocol_version)
        if not self.portHandler.openPort():
            raise Exception(f'Failed to open port {self.config.device_name}')

        if not self.portHandler.setBaudRate(self.config.baudrate):
            raise Exception(f'failed to set baudrate to {self.config.baudrate}')

        # self.operating_mode = OperatingMode.UNKNOWN
        # self.torque_enabled = False
        # self._disable_torque()

        self.operating_modes = [None for _ in range(32)]
        self.torque_enabled = [None for _ in range(32)]
        return True

    def disconnect(self):
        self.portHandler.closePort()

    def set_goal_position(self, motor_id, goal_position):
        # if self.operating_modes[motor_id] is not OperatingMode.POSITION:
        #     self._disable_torque(motor_id)
        #     self.set_operating_mode(motor_id, OperatingMode.POSITION)

        # if not self.torque_enabled[motor_id]:
        #     self._enable_torque(motor_id)

        # self._enable_torque(motor_id)
        dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(self.portHandler, motor_id,
                                                                       self.ADDR_GOAL_POSITION, goal_position)
        # self._process_response(dxl_comm_result, dxl_error)
        # print(f'set position of motor {motor_id} to {goal_position}')

    def set_pwm_value(self, motor_id: int, pwm_value, tries=3):
        if self.operating_modes[motor_id] is not OperatingMode.PWM:
            self._disable_torque(motor_id)
            self.set_operating_mode(motor_id, OperatingMode.PWM)

        if not self.torque_enabled[motor_id]:
            self._enable_torque(motor_id)
            # print(f'enabling torque')
        dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(self.portHandler, motor_id,
                                                                       self.ADDR_GOAL_PWM, pwm_value)
        # self._process_response(dxl_comm_result, dxl_error)
        # print(f'set pwm of motor {motor_id} to {pwm_value}')
        if dxl_comm_result != COMM_SUCCESS:
            if tries <= 1:
                raise ConnectionError(f"dxl_comm_result: {self.packetHandler.getTxRxResult(dxl_comm_result)}")
            else:
                print(f'dynamixel pwm setting failure trying again with {tries - 1} tries')
                self.set_pwm_value(motor_id, pwm_value, tries=tries - 1)
        elif dxl_error != 0:
            print(f'dxl error {dxl_error}')
            raise ConnectionError(f"dynamixel error: {self.packetHandler.getTxRxResult(dxl_error)}")

    def read_temperature(self, motor_id: int):
        return self._read_value(motor_id, ReadAttribute.TEMPERATURE, 1)

    def read_velocity(self, motor_id: int):
        pos = self._read_value(motor_id, ReadAttribute.VELOCITY, 4)
        if pos > 2 ** 31:
            pos -= 2 ** 32
        # print(f'read position {pos} for motor {motor_id}')
        return pos

    def read_position(self, motor_id: int):
        pos = self._read_value(motor_id, ReadAttribute.POSITION, 4)
        if pos > 2 ** 31:
            pos -= 2 ** 32
        # print(f'read position {pos} for motor {motor_id}')
        return pos

    def read_position_degrees(self, motor_id: int) -> float:
        return (self.read_position(motor_id) / 4096) * 360

    def read_position_radians(self, motor_id: int) -> float:
        return (self.read_position(motor_id) / 4096) * 2 * math.pi

    def read_current(self, motor_id: int):
        current = self._read_value(motor_id, ReadAttribute.CURRENT, 2)
        if current > 2 ** 15:
            current -= 2 ** 16
        return current

    def read_present_pwm(self, motor_id: int):
        return self._read_value(motor_id, ReadAttribute.PWM, 2)

    def read_hardware_error_status(self, motor_id: int):
        return self._read_value(motor_id, ReadAttribute.HARDWARE_ERROR_STATUS, 1)

    def disconnect(self):
        self.portHandler.closePort()

    def set_id(self, old_id, new_id, use_broadcast_id: bool = False):
        """
        sets the id of the dynamixel servo
        @param old_id: current id of the servo
        @param new_id: new id
        @param use_broadcast_id: set ids of all connected dynamixels if True.
         If False, change only servo with self.config.id
        @return:
        """
        if use_broadcast_id:
            current_id = 254
        else:
            current_id = old_id
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler,
                                                                       current_id, self.ADDR_ID, new_id)
        self._process_response(dxl_comm_result, dxl_error, old_id)
        self.config.id = id

    def _enable_torque(self, motor_id):
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, motor_id,
                                                                       self.ADDR_TORQUE_ENABLE, 1)
        self._process_response(dxl_comm_result, dxl_error, motor_id)
        self.torque_enabled[motor_id] = True

    def _disable_torque(self, motor_id):
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, motor_id,
                                                                       self.ADDR_TORQUE_ENABLE, 0)
        self._process_response(dxl_comm_result, dxl_error, motor_id)
        self.torque_enabled[motor_id] = False

    def _process_response(self, dxl_comm_result: int, dxl_error: int, motor_id: int):
        if dxl_comm_result != COMM_SUCCESS:
            raise ConnectionError(
                f"dxl_comm_result for motor {motor_id}: {self.packetHandler.getTxRxResult(dxl_comm_result)}")
        elif dxl_error != 0:
            print(f'dxl error {dxl_error}')
            raise ConnectionError(
                f"dynamixel error for motor {motor_id}: {self.packetHandler.getTxRxResult(dxl_error)}")

    def set_operating_mode(self, motor_id: int, operating_mode: OperatingMode):
        dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(self.portHandler, motor_id,
                                                                       self.OPERATING_MODE_ADDR, operating_mode.value)
        self._process_response(dxl_comm_result, dxl_error, motor_id)
        self.operating_modes[motor_id] = operating_mode

    def set_pwm_limit(self, motor_id: int, limit: int):
        dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(self.portHandler, motor_id,
                                                                       36, limit)
        self._process_response(dxl_comm_result, dxl_error, motor_id)

    def set_velocity_limit(self, motor_id: int, velocity_limit):
        dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(self.portHandler, motor_id,
                                                                       self.ADDR_VELOCITY_LIMIT, velocity_limit)
        self._process_response(dxl_comm_result, dxl_error, motor_id)

    def set_P(self, motor_id: int, P: int):
        dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(self.portHandler, motor_id,
                                                                       self.POSITION_P, P)
        self._process_response(dxl_comm_result, dxl_error, motor_id)

    def set_I(self, motor_id: int, I: int):
        dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(self.portHandler, motor_id,
                                                                       self.POSITION_I, I)
        self._process_response(dxl_comm_result, dxl_error, motor_id)

    def read_home_offset(self, motor_id: int):
        self._disable_torque(motor_id)
        # dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(self.portHandler, motor_id,
        #                                                                ReadAttribute.HOMING_OFFSET.value, home_position)
        home_offset = self._read_value(motor_id, ReadAttribute.HOMING_OFFSET, 4)
        # self._process_response(dxl_comm_result, dxl_error)
        self._enable_torque(motor_id)
        return home_offset

    def set_home_offset(self, motor_id: int, home_position: int):
        self._disable_torque(motor_id)
        dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(self.portHandler, motor_id,
                                                                       ReadAttribute.HOMING_OFFSET.value, home_position)
        self._process_response(dxl_comm_result, dxl_error, motor_id)
        self._enable_torque(motor_id)

    def set_baudrate(self, motor_id: int, baudrate):
        # translate baudrate into dynamixel baudrate setting id
        if baudrate == 57600:
            baudrate_id = 1
        elif baudrate == 1_000_000:
            baudrate_id = 3
        elif baudrate == 2_000_000:
            baudrate_id = 4
        elif baudrate == 3_000_000:
            baudrate_id = 5
        elif baudrate == 4_000_000:
            baudrate_id = 6
        else:
            raise Exception('baudrate not implemented')

        self._disable_torque(motor_id)
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, motor_id,
                                                                       ReadAttribute.BAUDRATE.value, baudrate_id)
        self._process_response(dxl_comm_result, dxl_error, motor_id)

    def _read_value(self, motor_id, attribute: ReadAttribute, num_bytes: int, tries=10):
        try:
            if num_bytes == 1:
                value, dxl_comm_result, dxl_error = self.packetHandler.read1ByteTxRx(self.portHandler,
                                                                                     motor_id,
                                                                                     attribute.value)
            elif num_bytes == 2:
                value, dxl_comm_result, dxl_error = self.packetHandler.read2ByteTxRx(self.portHandler,
                                                                                     motor_id,
                                                                                     attribute.value)
            elif num_bytes == 4:
                value, dxl_comm_result, dxl_error = self.packetHandler.read4ByteTxRx(self.portHandler,
                                                                                     motor_id,
                                                                                     attribute.value)
        except Exception:
            if tries == 0:
                raise Exception
            else:
                return self._read_value(motor_id, attribute, num_bytes, tries=tries - 1)
        if dxl_comm_result != COMM_SUCCESS:
            if tries <= 1:
                # print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
                raise ConnectionError(f'dxl_comm_result {dxl_comm_result} for servo {motor_id} value {value}')
            else:
                print(f'dynamixel read failure for servo {motor_id} trying again with {tries - 1} tries')
                time.sleep(0.02)
                return self._read_value(motor_id, attribute, num_bytes, tries=tries - 1)
        elif dxl_error != 0:  # # print("%s" % self.packetHandler.getRxPacketError(dxl_error))
            # raise ConnectionError(f'dxl_error {dxl_error} binary ' + "{0:b}".format(37))
            if tries == 0 and dxl_error != 128:
                raise Exception(f'Failed to read value from motor {motor_id} error is {dxl_error}')
            else:
                return self._read_value(motor_id, attribute, num_bytes, tries=tries - 1)
        return value

    def set_home_position(self, motor_id: int):
        print(f'setting home position for motor {motor_id}')
        self.set_home_offset(motor_id, 0)
        current_position = self.read_position(motor_id)
        print(f'position before {current_position}')
        self.set_home_offset(motor_id, -current_position)
        # dynamixel.set_home_offset(motor_id, -4096)
        # dynamixel.set_home_offset(motor_id, -4294964109)
        current_position = self.read_position(motor_id)
        # print(f'signed position {current_position - 2** 32}')
        print(f'position after {current_position}')

class MotorControlType(Enum):
    PWM = auto()
    POSITION_CONTROL = auto()
    DISABLED = auto()
    UNKNOWN = auto()

class Robot:
    # def __init__(self, device_name: str, baudrate=1_000_000, servo_ids=[1, 2, 3, 4, 5]):
    def __init__(self, dynamixel, baudrate=1_000_000, servo_ids=[1, 2, 3, 4, 5]):
        self.servo_ids = servo_ids
        self.dynamixel = dynamixel
        # self.dynamixel = Dynamixel.Config(baudrate=baudrate, device_name=device_name).instantiate()
        self.position_reader = GroupSyncRead(
            self.dynamixel.portHandler,
            self.dynamixel.packetHandler,
            ReadAttribute.POSITION.value,
            4)
        for id in self.servo_ids:
            self.position_reader.addParam(id)

        self.velocity_reader = GroupSyncRead(
            self.dynamixel.portHandler,
            self.dynamixel.packetHandler,
            ReadAttribute.VELOCITY.value,
            4)
        for id in self.servo_ids:
            self.velocity_reader.addParam(id)

        self.pos_writer = GroupSyncWrite(
            self.dynamixel.portHandler,
            self.dynamixel.packetHandler,
            self.dynamixel.ADDR_GOAL_POSITION,
            4)
        for id in self.servo_ids:
            self.pos_writer.addParam(id, [2048])

        self.pwm_writer = GroupSyncWrite(
            self.dynamixel.portHandler,
            self.dynamixel.packetHandler,
            self.dynamixel.ADDR_GOAL_PWM,
            2)
        for id in self.servo_ids:
            self.pwm_writer.addParam(id, [2048])
        self._disable_torque()
        self.motor_control_state = MotorControlType.DISABLED

    def read_position(self, tries=2):
        """
        Reads the joint positions of the robot. 2048 is the center position. 0 and 4096 are 180 degrees in each direction.
        :param tries: maximum number of tries to read the position
        :return: list of joint positions in range [0, 4096]
        """
        result = self.position_reader.txRxPacket()
        if result != 0:
            if tries > 0:
                return self.read_position(tries=tries - 1)
            else:
                print(f'failed to read position!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        positions = []
        for id in self.servo_ids:
            position = self.position_reader.getData(id, ReadAttribute.POSITION.value, 4)
            if position > 2 ** 31:
                position -= 2 ** 32
            positions.append(position)
        return positions

    def read_velocity(self):
        """
        Reads the joint velocities of the robot.
        :return: list of joint velocities,
        """
        self.velocity_reader.txRxPacket()
        velocties = []
        for id in self.servo_ids:
            velocity = self.velocity_reader.getData(id, ReadAttribute.VELOCITY.value, 4)
            if velocity > 2 ** 31:
                velocity -= 2 ** 32
            velocties.append(velocity)
        return velocties

    @staticmethod
    def _degrees_to_pos(degrees: float) -> int:
        degrees = max(-180, min(180, degrees))
        abs_pos = round(abs(degrees) / 180 * 2048)
        return 2048 + (abs_pos if degrees >= 0 else -abs_pos)

    def set_goal_angle(self, degrees: List[float]):
        self.set_goal_pos(action=[ self._degrees_to_pos(motor_degrees) for motor_degrees in degrees ])

    def set_goal_pos(self, action):
        """

        :param action: list or numpy array of target joint positions in range [0, 4096]
        """
        if not self.motor_control_state is MotorControlType.POSITION_CONTROL:
            self._set_position_control()
        for i, motor_id in enumerate(self.servo_ids):
            data_write = [DXL_LOBYTE(DXL_LOWORD(action[i])),
                          DXL_HIBYTE(DXL_LOWORD(action[i])),
                          DXL_LOBYTE(DXL_HIWORD(action[i])),
                          DXL_HIBYTE(DXL_HIWORD(action[i]))]
            self.pos_writer.changeParam(motor_id, data_write)

        self.pos_writer.txPacket()

    def set_pwm(self, action):
        """
        Sets the pwm values for the servos.
        :param action: list or numpy array of pwm values in range [0, 885]
        """
        if not self.motor_control_state is MotorControlType.PWM:
            self._set_pwm_control()
        for i, motor_id in enumerate(self.servo_ids):
            data_write = [DXL_LOBYTE(DXL_LOWORD(action[i])),
                          DXL_HIBYTE(DXL_LOWORD(action[i])),
                          ]
            self.pwm_writer.changeParam(motor_id, data_write)

        self.pwm_writer.txPacket()

    def set_trigger_torque(self):
        """
        Sets a constant torque torque for the last servo in the chain. This is useful for the trigger of the leader arm
        """
        self.dynamixel._enable_torque(self.servo_ids[-1])
        self.dynamixel.set_pwm_value(self.servo_ids[-1], 200)

    def limit_pwm(self, limit: Union[int, list, np.ndarray]):
        """
        Limits the pwm values for the servos in for position control
        @param limit: 0 ~ 885
        @return:
        """
        if isinstance(limit, int):
            limits = [limit, ] * 5
        else:
            limits = limit
        self._disable_torque()
        for motor_id, limit in zip(self.servo_ids, limits):
            self.dynamixel.set_pwm_limit(motor_id, limit)
        self._enable_torque()

    def _disable_torque(self):
        print(f'disabling torque for servos {self.servo_ids}')
        for motor_id in self.servo_ids:
            self.dynamixel._disable_torque(motor_id)

    def _enable_torque(self):
        print(f'enabling torque for servos {self.servo_ids}')
        for motor_id in self.servo_ids:
            self.dynamixel._enable_torque(motor_id)

    def _set_pwm_control(self):
        self._disable_torque()
        for motor_id in self.servo_ids:
            self.dynamixel.set_operating_mode(motor_id, OperatingMode.PWM)
        self._enable_torque()
        self.motor_control_state = MotorControlType.PWM

    def _set_position_control(self):
        self._disable_torque()
        for motor_id in self.servo_ids:
            self.dynamixel.set_operating_mode(motor_id, OperatingMode.POSITION)
        self._enable_torque()
        self.motor_control_state = MotorControlType.POSITION_CONTROL



def serial_ports():
    """ Lists serial port names

        :raises EnvironmentError:
            On unsupported or unknown platforms
        :returns:
            A list of the serial ports available on the system
    """
    if sys.platform.startswith('win'):
        ports = ['COM%s' % (i + 1) for i in range(256)]
    elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
        # this excludes your current terminal "/dev/tty"
        ports = glob.glob('/dev/tty[A-Za-z]*')
    elif sys.platform.startswith('darwin'):
        ports = glob.glob('/dev/tty.*')
    else:
        raise EnvironmentError('Unsupported platform')

    result = []
    for port in ports:
        try:
            s = serial.Serial(port)
            s.close()
            result.append(port)
        except (OSError, serial.SerialException):
            pass
    return result

def find_serial_port(port_pattern: str) -> str | None:
    # If port contains wildcard characters, we need to compare against all ports, otherwise we just
    # return the port the user specified
    if "*" in port_pattern or "?" in port_pattern or "[" in port_pattern:
        import fnmatch
        ports = serial_ports()
        matches = [port for port in ports if fnmatch.fnmatch(name=port, pat=port_pattern)]
        if len(matches) == 0:
            print("Error: No matching ports found")
            return None
        if len(matches) > 1:
            print(f"Error: Multiple ports match given pattern: {', '.join(ports)}")
            return None
        return matches[0]
    else:
        return port_pattern

if __name__ == "__main__":
    parser = argparse.ArgumentParser("robotest")
    parser.add_argument("--list-ports", action="store_true", help="List available serial ports")
    parser.add_argument("--port", action="store", type=str, help="Serial port to use")
    options = parser.parse_args()

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
    print(f"Using {port}")

    dynamixel = Dynamixel.Config(
        baudrate=57600,
        device_name=port,
    ).instantiate()

    motor_id = 1
    pos = dynamixel.read_position(motor_id)
    for i in range(10):
        s = time.monotonic()
        pos = dynamixel.read_position(motor_id)
        delta = time.monotonic() - s
        print(f'read position took {delta}')
        print(f'position {pos}')

    print(f"Motor 1: {dynamixel.read_position_degrees(motor_id=motor_id)} deg")


    robot = Robot(dynamixel=dynamixel, baudrate=57600, servo_ids=[1, 2, 3, 4])
    # robot.set_goal_pos([2048,2048,2048,2048])
    # #while robot.read_position()[0] != 2048:
    # #    pass
    # time.sleep(2)
    # print(robot.read_position())
    # robot._disable_torque()

    # Reset to home position
    robot.set_goal_angle(degrees=[0,0,0,0])
    time.sleep(2)
    print(robot.read_position())
    #robot._disable_torque()
    #exit()

    # Target IK
    target = [-120, 150, 50]
    my_chain = Chain.from_urdf_file("robot-arm.urdf", active_links_mask=[ False, False, True, True, False, True, True, False ])
    motor_id_by_joint_idx = { 2: 1, 3: 2, 5: 3, 6: 4 }
    joint_idx_by_motor_id = { motor_id: joint_idx for joint_idx, motor_id in motor_id_by_joint_idx.items() }
    joint_angles = my_chain.inverse_kinematics(target_position=target)
    print([ np.rad2deg(angle) for angle in joint_angles ])
    joint_degrees = []
    for motor_id in sorted(joint_idx_by_motor_id.keys()):
        joint_idx = joint_idx_by_motor_id[motor_id]
        angle = joint_angles[joint_idx]
        joint_degrees.append(np.rad2deg(angle))
    joint_degrees[1] = -joint_degrees[1]
    print(joint_degrees)
    robot.set_goal_angle(degrees=joint_degrees)
    time.sleep(3)
    robot._disable_torque()



