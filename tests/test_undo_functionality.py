#!/usr/bin/env python3
"""
Test script to verify that the undo functionality works correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from .utils.box_editor import Box, BoxEditor
import tkinter as tk

def test_undo_functionality():
    """Test the undo functionality with various operations."""
    print("Testing Undo Functionality")
    print("=" * 40)
    
    # Create a test window and canvas
    root = tk.Tk()
    root.withdraw()  # Hide the window
    canvas = tk.Canvas(root, width=400, height=400)
    
    # Create box editor
    editor = BoxEditor(canvas)
    
    print("Initial state:")
    boxes, labels = editor.get_boxes()
    print(f"Boxes: {len(boxes)}, History size: {len(editor.history_stack)}")
    
    # Test 1: Add some boxes
    print("\nTest 1: Adding boxes")
    editor.set_boxes([[50, 50, 150, 150], [200, 200, 300, 300]], ["box1", "box2"])
    boxes, labels = editor.get_boxes()
    print(f"After adding boxes: {len(boxes)} boxes, History size: {len(editor.history_stack)}")
    print(f"Box coordinates: {boxes}")
    print(f"Labels: {labels}")
    
    # Test 2: Undo the box addition
    print("\nTest 2: Undo box addition")
    result = editor.undo()
    boxes, labels = editor.get_boxes()
    print(f"Undo result: {result}")
    print(f"After undo: {len(boxes)} boxes, History size: {len(editor.history_stack)}")
    
    # Test 3: Add boxes again and modify them
    print("\nTest 3: Add boxes and modify")
    editor.set_boxes([[50, 50, 150, 150]], ["test"])
    boxes, labels = editor.get_boxes()
    print(f"Added 1 box: {boxes[0]}")
    
    # Simulate a move operation (save to history then modify)
    editor._save_to_history()
    editor.boxes[0].move_by(10, 10)
    editor._redraw_all()
    
    boxes, labels = editor.get_boxes()
    print(f"After moving: {boxes[0]}")
    print(f"History size: {len(editor.history_stack)}")
    
    # Test 4: Undo the move
    print("\nTest 4: Undo the move")
    result = editor.undo()
    boxes, labels = editor.get_boxes()
    print(f"Undo result: {result}")
    print(f"After undo move: {boxes[0]}")
    print(f"History size: {len(editor.history_stack)}")
    
    # Test 5: Try to undo when no history
    print("\nTest 5: Multiple undos until empty")
    while editor.history_stack:
        result = editor.undo()
        print(f"Undo result: {result}, remaining history: {len(editor.history_stack)}")
    
    # Try one more undo
    result = editor.undo()
    print(f"Undo when empty: {result}")
    
    root.destroy()
    print("\nUndo functionality test completed!")

if __name__ == "__main__":
    test_undo_functionality() 