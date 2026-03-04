"""
Barrett Hand (BH282) interface for real hardware control.

The Barrett Hand is a 3-finger dexterous hand with 4 DOF per finger plus spread.
This interface provides control via Ethernet/IP or CAN bus.
"""

import numpy as np
from typing import Optional, List, Tuple
import time
import logging
import socket

logger = logging.getLogger(__name__)


class BarrettHand:
    """Interface for Barrett BH282 Hand control."""
    
    def __init__(
        self,
        ip_address: str = "192.168.1.103",
        port: int = 10000,
        timeout: float = 1.0,
    ):
        """
        Initialize Barrett Hand interface.
        
        Args:
            ip_address: Hand IP address (Ethernet connection)
            port: TCP port (default: 10000)
            timeout: Connection timeout in seconds
        """
        self.ip_address = ip_address
        self.port = port
        self.timeout = timeout
        self.connected = False
        self._socket = None
        
        # Barrett Hand has 3 fingers, each with 4 joints
        # Plus spread joint, total 13 joints
        self.num_fingers = 3
        self.joints_per_finger = 4
        self.num_joints = 13
        
    def connect(self) -> bool:
        """
        Connect to the Barrett Hand.
        
        Returns:
            True if connection successful
        """
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.settimeout(self.timeout)
            self._socket.connect((self.ip_address, self.port))
            self.connected = True
            logger.info(f"Connected to Barrett Hand at {self.ip_address}:{self.port}")
            
            # Initialize hand (home position)
            self.initialize()
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Barrett Hand: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from the Barrett Hand."""
        if self._socket is not None:
            try:
                self._socket.close()
            except:
                pass
        self.connected = False
        logger.info("Disconnected from Barrett Hand")
    
    def initialize(self) -> bool:
        """
        Initialize the hand (move to home position).
        
        Returns:
            True if initialization successful
        """
        if not self.connected:
            raise RuntimeError("Hand not connected")
        
        try:
            # Send initialization command
            # Barrett Hand initialization sequence
            cmd = "INIT\n"
            self._socket.send(cmd.encode())
            time.sleep(2.0)  # Wait for initialization
            
            logger.info("Barrett Hand initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize hand: {e}")
            return False
    
    def set_finger_positions(
        self,
        finger_positions: List[float],
        spread: Optional[float] = None,
        speed: float = 0.5,
    ) -> bool:
        """
        Set finger positions.
        
        Args:
            finger_positions: List of 3 finger positions (0 = open, 1 = closed)
            spread: Spread angle in radians (optional)
            speed: Movement speed (0-1)
        
        Returns:
            True if command sent successfully
        """
        if not self.connected:
            raise RuntimeError("Hand not connected")
        
        if len(finger_positions) != self.num_fingers:
            raise ValueError(f"Expected {self.num_fingers} finger positions, got {len(finger_positions)}")
        
        try:
            # Convert normalized positions to joint angles
            # Barrett Hand uses joint angles in radians
            # Simplified model: each finger has 2 main joints
            cmd_parts = []
            
            for i, pos in enumerate(finger_positions):
                # Clamp position
                pos = np.clip(pos, 0.0, 1.0)
                
                # Convert to joint angles (simplified model)
                # Finger 1: spread joint + 2 main joints
                # Finger 2: spread joint + 2 main joints  
                # Finger 3: spread joint + 2 main joints
                # Spread: separate joint
                
                # Main finger joints (simplified: 0 to 1.4 radians)
                joint1 = pos * 1.4  # First joint
                joint2 = pos * 1.2  # Second joint
                
                cmd_parts.append(f"F{i+1}J1={joint1:.3f}")
                cmd_parts.append(f"F{i+1}J2={joint2:.3f}")
            
            # Set spread if provided
            if spread is not None:
                spread = np.clip(spread, 0.0, 2.0)  # Spread limits
                cmd_parts.append(f"SPREAD={spread:.3f}")
            
            # Send command
            cmd = " ".join(cmd_parts) + "\n"
            self._socket.send(cmd.encode())
            time.sleep(0.1)
            
            return True
        except Exception as e:
            logger.error(f"Failed to set finger positions: {e}")
            return False
    
    def set_open_amount(self, amount: float, speed: float = 0.5) -> bool:
        """
        Set hand open amount (0 = closed, 1 = open).
        
        Args:
            amount: Open amount (0.0 = closed, 1.0 = open)
            speed: Movement speed (0-1)
        
        Returns:
            True if command sent successfully
        """
        # For simplified control, set all fingers to same position
        finger_positions = [1.0 - amount] * self.num_fingers  # Invert: 0=open, 1=closed
        return self.set_finger_positions(finger_positions, speed=speed)
    
    def open(self, speed: float = 0.5) -> bool:
        """
        Open the hand (all fingers extended).
        
        Args:
            speed: Movement speed (0-1)
        
        Returns:
            True if command sent successfully
        """
        return self.set_open_amount(1.0, speed=speed)
    
    def close(self, speed: float = 0.5) -> bool:
        """
        Close the hand (all fingers closed).
        
        Args:
            speed: Movement speed (0-1)
        
        Returns:
            True if command sent successfully
        """
        return self.set_open_amount(0.0, speed=speed)
    
    def get_status(self) -> dict:
        """
        Get hand status.
        
        Returns:
            Dictionary with status information
        """
        if not self.connected:
            raise RuntimeError("Hand not connected")
        
        try:
            # Request status
            cmd = "STATUS\n"
            self._socket.send(cmd.encode())
            time.sleep(0.1)
            
            # Read response
            response = self._socket.recv(1024).decode()
            
            # Parse response (format depends on Barrett Hand model)
            # Simplified parsing
            status = {
                'connected': True,
                'fingers': [0.5] * self.num_fingers,  # Placeholder
                'spread': 0.0,
            }
            
            return status
        except Exception as e:
            logger.error(f"Failed to get hand status: {e}")
            return {'error': str(e)}
    
    def get_open_amount(self) -> float:
        """
        Get current open amount (0 = closed, 1 = open).
        
        Returns:
            Open amount (0.0-1.0)
        """
        status = self.get_status()
        # Average finger positions (simplified)
        if 'fingers' in status:
            avg_pos = np.mean(status['fingers'])
            return 1.0 - avg_pos  # Invert: 0=open, 1=closed
        return 0.5

