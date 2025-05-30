#!/usr/bin/env python3
"""
Test script to verify that box editing constraints work correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.box_editor import Box

def test_box_constraints():
    """Test box movement and resizing with boundary constraints."""
    print("Testing Box Movement and Resizing Constraints")
    print("=" * 50)
    
    # Create a test box
    box = Box(50, 50, 150, 150, "test")
    print(f"Initial box: ({box.x1}, {box.y1}, {box.x2}, {box.y2})")
    
    # Define image boundaries (left, top, right, bottom)
    image_bounds = (0, 0, 200, 200)
    print(f"Image bounds: {image_bounds}")
    
    # Test 1: Move box within bounds
    print("\nTest 1: Move box within bounds")
    box.move_by(10, 10, bounds=image_bounds)
    print(f"After moving by (10, 10): ({box.x1}, {box.y1}, {box.x2}, {box.y2})")
    
    # Test 2: Try to move box outside left boundary
    print("\nTest 2: Try to move box outside left boundary")
    box.move_by(-100, 0, bounds=image_bounds)
    print(f"After moving by (-100, 0): ({box.x1}, {box.y1}, {box.x2}, {box.y2})")
    
    # Test 3: Try to move box outside right boundary
    print("\nTest 3: Try to move box outside right boundary")
    box.move_by(200, 0, bounds=image_bounds)
    print(f"After moving by (200, 0): ({box.x1}, {box.y1}, {box.x2}, {box.y2})")
    
    # Reset box position
    box.x1, box.y1, box.x2, box.y2 = 50, 50, 150, 150
    print(f"\nReset box: ({box.x1}, {box.y1}, {box.x2}, {box.y2})")
    
    # Test 4: Try to resize box outside bounds (southeast handle)
    print("\nTest 4: Try to resize box outside bounds (southeast handle)")
    box.resize_by('se', 100, 100, bounds=image_bounds)
    print(f"After resizing se by (100, 100): ({box.x1}, {box.y1}, {box.x2}, {box.y2})")
    
    # Test 5: Try to resize box outside bounds (northwest handle)
    print("\nTest 5: Try to resize box outside bounds (northwest handle)")
    box.resize_by('nw', -100, -100, bounds=image_bounds)
    print(f"After resizing nw by (-100, -100): ({box.x1}, {box.y1}, {box.x2}, {box.y2})")
    
    print("\nTest completed! Boxes should be constrained within bounds (0, 0, 200, 200)")

if __name__ == "__main__":
    test_box_constraints() 