"""
gfvla_env_dino.py

GF-VLA Environment DINOv2 model for point cloud processing.
This is a compatibility alias - the actual implementation is in dualarmvla_dino.py
"""

# Temporary compatibility import
# This allows imports to work before the file is renamed
# Compatibility import from dualarmvla_dino
from .dualarmvla_dino import GfvlaEnvDinov2

__all__ = ['GfvlaEnvDinov2']

