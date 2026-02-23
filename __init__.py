"""
ComfyUI Custom Node: Compressed Sensing Reconstruction
======================================================
Install: copy this folder into ComfyUI/custom_nodes/
"""

from .nodes import CompressedSensingNode

NODE_CLASS_MAPPINGS = {
    "CompressedSensing": CompressedSensingNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CompressedSensing": "Compressed Sensing (10% Random Sampling)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
