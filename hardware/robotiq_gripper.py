"""
Robotiq Gripper interface for real hardware control.

Supports Robotiq 2F-85, 2F-140, and similar models via Modbus RTU or Ethernet.
"""

import numpy as np
from typing import Optional
import time
import logging
import serial

logger = logging.getLogger(__name__)


class RobotiqGripper:
    """Interface for Robotiq Gripper control."""
    
    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        baudrate: int = 115200,
        timeout: float = 1.0,
    ):
        """
        Initialize Robotiq Gripper interface.
        
        Args:
            port: Serial port (Linux: /dev/ttyUSB0, Windows: COM3)
            baudrate: Serial baudrate (default: 115200)
            timeout: Serial timeout in seconds
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.connected = False
        self._serial = None
        self.min_position = 0.0  # Closed (mm)
        self.max_position = 85.0  # Open (mm)
        
    def connect(self) -> bool:
        """
        Connect to the gripper.
        
        Returns:
            True if connection successful
        """
        try:
            self._serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                bytesize=8,
                parity=serial.PARITY_NONE,
                stopbits=1,
            )
            self.connected = True
            logger.info(f"Connected to Robotiq gripper on {self.port}")
            
            # Activate gripper (required for Robotiq)
            self.activate()
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Robotiq gripper: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from the gripper."""
        if self._serial is not None and self._serial.is_open:
            self._serial.close()
        self.connected = False
        logger.info("Disconnected from Robotiq gripper")
    
    def activate(self) -> bool:
        """
        Activate the gripper (required before use).
        
        Returns:
            True if activation successful
        """
        if not self.connected:
            raise RuntimeError("Gripper not connected")
        
        try:
            # Robotiq activation sequence
            # Send activation command (varies by model)
            activation_cmd = bytes([0x09, 0x10, 0x03, 0xE8, 0x00, 0x03, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
            self._serial.write(activation_cmd)
            time.sleep(2.0)  # Wait for activation
            
            # Check activation status
            status = self.get_status()
            if status.get('activated', False):
                logger.info("Robotiq gripper activated")
                return True
            else:
                logger.warning("Robotiq gripper activation may have failed")
                return False
        except Exception as e:
            logger.error(f"Failed to activate gripper: {e}")
            return False
    
    def set_position(self, position: float, speed: int = 255, force: int = 255) -> bool:
        """
        Set gripper position.
        
        Args:
            position: Position in mm (0 = closed, 85 = open for 2F-85)
            speed: Speed (0-255)
            force: Force (0-255)
        
        Returns:
            True if command sent successfully
        """
        if not self.connected:
            raise RuntimeError("Gripper not connected")
        
        # Clamp position to valid range
        position = np.clip(position, self.min_position, self.max_position)
        position_int = int(position * 10)  # Convert to 0.1mm units
        
        try:
            # Robotiq position command (Modbus RTU)
            # Format: [slave_id, function_code, address_high, address_low, 
            #          value_high, value_low, speed, force]
            cmd = bytes([
                0x09,  # Slave ID
                0x10,  # Function code (write multiple registers)
                0x03, 0xE8,  # Starting address
                0x00, 0x03,  # Number of registers
                0x06,  # Byte count
                0x09, 0x00,  # Action request (0x0900 = go to position)
                (position_int >> 8) & 0xFF, position_int & 0xFF,  # Position
                speed,  # Speed
                force,  # Force
            ])
            
            self._serial.write(cmd)
            time.sleep(0.1)  # Small delay for command processing
            return True
        except Exception as e:
            logger.error(f"Failed to set gripper position: {e}")
            return False
    
    def set_open_amount(self, amount: float, speed: int = 255, force: int = 255) -> bool:
        """
        Set gripper open amount (0 = closed, 1 = open).
        
        Args:
            amount: Open amount (0.0 = closed, 1.0 = open)
            speed: Speed (0-255)
            force: Force (0-255)
        
        Returns:
            True if command sent successfully
        """
        position = amount * self.max_position
        return self.set_position(position, speed, force)
    
    def open(self, speed: int = 255) -> bool:
        """
        Open the gripper.
        
        Args:
            speed: Speed (0-255)
        
        Returns:
            True if command sent successfully
        """
        return self.set_open_amount(1.0, speed=speed)
    
    def close(self, speed: int = 255, force: int = 255) -> bool:
        """
        Close the gripper.
        
        Args:
            speed: Speed (0-255)
            force: Force (0-255)
        
        Returns:
            True if command sent successfully
        """
        return self.set_open_amount(0.0, speed=speed, force=force)
    
    def get_status(self) -> dict:
        """
        Get gripper status.
        
        Returns:
            Dictionary with status information
        """
        if not self.connected:
            raise RuntimeError("Gripper not connected")
        
        try:
            # Read status registers
            read_cmd = bytes([0x09, 0x03, 0x07, 0xD0, 0x00, 0x03])
            self._serial.write(read_cmd)
            time.sleep(0.1)
            
            response = self._serial.read(9)  # Expected response length
            
            if len(response) >= 9:
                # Parse response
                gOBJ = (response[3] >> 6) & 0x03  # Object detection
                gSTA = (response[3] >> 4) & 0x03  # Gripper status
                gACT = (response[3] >> 0) & 0x01  # Activation status
                
                position_raw = (response[5] << 8) | response[6]
                position = position_raw / 10.0  # Convert from 0.1mm to mm
                
                return {
                    'activated': gACT == 1,
                    'status': gSTA,
                    'object_detected': gOBJ != 0,
                    'position': position,
                    'position_normalized': position / self.max_position,
                }
            else:
                return {'error': 'Invalid response'}
        except Exception as e:
            logger.error(f"Failed to get gripper status: {e}")
            return {'error': str(e)}
    
    def get_open_amount(self) -> float:
        """
        Get current open amount (0 = closed, 1 = open).
        
        Returns:
            Open amount (0.0-1.0)
        """
        status = self.get_status()
        return status.get('position_normalized', 0.5)

